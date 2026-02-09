from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
from AB3DMOT.AB3DMOT_libs.box import Box3D
from AB3DMOT.AB3DMOT_libs.matching import data_association
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import print_log
import numpy as np, os, copy, math

from DMSTrack.differentiable_kalman_filter import DKF
import torch
import torch.nn as nn
from DMSTrack.loss import get_2d_center_distance_matrix, get_association_loss, get_neg_log_likelihood_loss
import time
from YJQTrack.CI.covariance_intersection import CovarianceIntersection
from YJQTrack.CI.covariance_intersection import FastCovarianceIntersection
from YJQTrack.model import DMSTrack
from YJQTrack.q_noise_enhanced.q_noise_with_innovation import InnovationBasedQNet, InnovationBasedQLoss
from YJQTrack.q_noise_enhanced.q_noise_with_physics import PhysicsConstrainedQNet, PhysicsConstrainedQLoss
from YJQTrack.q_noise_enhanced.q_noise_with_MSTA import MultiScaleAttentionQNet, MultiScaleAttentionQLoss
from YJQTrack.q_noise_enhanced.q_noise_with_trajectory import TrajectoryBasedQNet, TrajectoryBasedQLoss
from YJQTrack.q_noise_enhanced.track_history_manager import TrackHistoryManager
from typing import Dict, List, Optional, Tuple, Union, Any
import torch.nn.functional as F
from YJQTrack.q_noise_enhanced.full_q_for_ral_r1.q_noise_with_trajectory import SimplifiedQNet_Diagonal
from YJQTrack.q_noise_enhanced.full_q_for_ral_r1.q_noise_with_trajectory import SimplifiedQNet_FullCov

class EnhancedProcessNoiseManager:
    """
    Enhanced process noise manager
    Integrates multiple Q matrix learning methods
    """

    def __init__(self, config, device, dtype):
        self.device = device
        self.dtype = dtype
        self.dim_x = config['dim_x']
        self.config = config

        # Get default Q matrix

        # Innovation sequence history cache
        self.innovation_histories = {}  # track_id -> innovation_history
        self.max_history_length = 20

    def update_innovation_history(self, track_id: int, innovation: torch.Tensor):
        """Update innovation sequence history"""
        if track_id not in self.innovation_histories:
            self.innovation_histories[track_id] = []

        self.innovation_histories[track_id].append(innovation.detach().clone())

        # Maintain fixed length
        if len(self.innovation_histories[track_id]) > self.max_history_length:
            self.innovation_histories[track_id].pop(0)
        return self.innovation_histories

    def get_innovation_sequence(self, track_id: int, seq_len: int = 10) -> Optional[torch.Tensor]:
        """Get innovation sequence"""
        if track_id not in self.innovation_histories:
            return None

        history = self.innovation_histories[track_id]
        if len(history) < seq_len:
            return None

        # Get the most recent seq_len innovation vectors
        recent_innovations = history[-seq_len:]
        return torch.stack(recent_innovations, dim=0)  # [seq_len, dim_z]

    def compute_adaptive_Q(self, valid_length, track_id: int, state_history: torch.Tensor,
                           current_innovation: Optional[torch.Tensor] = None,
                           positional_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive Q matrix

        Args:
            track_id: Track ID
            state_history: [seq_len, dim_x] State history
            current_innovation: [dim_z] Current innovation vector
            positional_features: [18] Positional features

        Returns:
            adaptive_Q: [dim_x, dim_x] Adaptive Q matrix
        """
        # if state_history.shape[0] < 5:  # History not long enough
        if len(state_history) < 5:  # History not long enough
            return self.default_Q

        try:
            # list -> tensor
            if not isinstance(state_history, torch.Tensor):
                state_history = torch.stack([torch.tensor(s, dtype=self.dtype, device=self.device)
                                             for s in state_history])
            # Prepare input data
            state_batch = state_history.unsqueeze(0)  # [1, seq_len, dim_x]

            if self.method_type == 'innovation_based':
                innovation_seq = self.get_innovation_sequence(track_id, seq_len=10)

                if innovation_seq is None:
                    return self.default_Q
                # list -> tensor
                if not isinstance(innovation_seq, torch.Tensor):
                    innovation_seq = torch.stack([torch.tensor(s, dtype=self.dtype, device=self.device)
                                                  for s in innovation_seq])

                innovation_batch = innovation_seq.unsqueeze(0)  # [1, seq_len, dim_z]
                current_state_batch = state_history[-1:].unsqueeze(0)  # [1, dim_x]

                if current_innovation is not None:
                    current_innovation_batch = current_innovation.unsqueeze(0)  # [1, dim_z]
                else:
                    current_innovation_batch = torch.zeros(1, 7, dtype=self.dtype, device=self.device)

                # Network forward propagation
                adaptive_Q_batch, adaptive_weight = self.q_net(
                    innovation_batch, state_batch, current_state_batch, current_innovation_batch
                )
                adaptive_Q = adaptive_Q_batch[0]
                # Blend default Q and adaptive Q
                # adaptive_Q = self.q_net.get_blended_Q(
                #     self.default_Q, adaptive_Q_batch, adaptive_weight
                # )[0]

            elif self.method_type == 'physics_constrained':
                if positional_features is None:
                    positional_features = torch.zeros(18, dtype=self.dtype, device=self.device)

                positional_batch = positional_features.unsqueeze(0)  # [1, 18]

                adaptive_Q_batch, motion_probs, uncertainty_level = self.q_net(
                    state_batch, positional_batch
                )
                adaptive_Q = adaptive_Q_batch[0]

            elif self.method_type == 'multiscale_attention':
                adaptive_Q_batch, temporal_weights, structure_masks, scale_weights = self.q_net(
                    state_batch
                )

                # Blend using time-varying weights
                weight = temporal_weights[0, 0].item()
                adaptive_Q = (1 - weight) * self.default_Q + weight * adaptive_Q_batch[0]

            else:
                return self.default_Q

            return adaptive_Q

        except Exception as e:
            print(f"Error in adaptive Q computation for track {track_id}: {e}")
            return self.default_Q


class BestForNowTrack(DMSTrack):
    def __init__(self, cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=None, ID_init=0,
                 dtype=None, device=None, differentiable_kalman_filter_config=None,
                 observation_covariance_net_dict=None, force_gt_as_predicted_track=False, use_static_default_R=False,
                 use_multiple_nets=False, enable_learnable_Q=False,delay_config=None, q_learning_config=None, q_net_dict=None):
        super().__init__(cfg, cat, calib, oxts, img_dir, vis_dir, hw, log, ID_init=0,
                         dtype=dtype, device=device,
                         differentiable_kalman_filter_config=differentiable_kalman_filter_config,
                         observation_covariance_net_dict=observation_covariance_net_dict,
                         force_gt_as_predicted_track=force_gt_as_predicted_track,
                         use_static_default_R=use_static_default_R,
                         use_multiple_nets=use_multiple_nets, enable_learnable_Q=enable_learnable_Q)

        # self.covariance_intersection = CovarianceIntersection(dtype=dtype, device=device)
        self.covariance_intersection = FastCovarianceIntersection(
            dtype=dtype,
            device=device,
            method='improved_trace'  # Recommended to use improved trace method
        )

        # self.q_net_dict = q_net_dict
        self.q_manager = EnhancedProcessNoiseManager(q_learning_config, self.device, self.dtype)
        self.innovation_buffer = {}
        self.window_size = 20
        self.min_samples = 3
        self.s_theoretical = {}
        self.s_empirical = {}
        self.q_loss_fn = TrajectoryBasedQLoss()
        self.innovations_id_list = []
        self.is_training = True
        self.q_usage_stats = {
            'frames_processed': 0,
            'total_predictions': 0,
            'adaptive_q_used': 0,
            'default_q_used': 0,
            'per_frame_stats': []
        }
        self.delay_config = delay_config if delay_config else {}
        self.detection_buffer = {}
        # Maximum buffer length (keep recent N frames)
        self.max_buffer_size = 10

    def add_detection_to_buffer(self, cav_id, frame, dets, info, feature, transformation_matrix):
        """Add detection data to buffer"""
        if cav_id not in self.detection_buffer:
            self.detection_buffer[cav_id] = {}

        self.detection_buffer[cav_id][frame] = {
            'dets': dets,
            'info': info,
            'feature': feature,
            'transformation_matrix': transformation_matrix
        }

        # Clean old buffer data
        self._clean_old_buffer(cav_id, frame)

    def _clean_old_buffer(self, cav_id, current_frame):
        """Clean data older than buffer size"""
        frames_to_keep = list(range(
            max(0, current_frame - self.max_buffer_size + 1),
            current_frame + 1
        ))

        frames_in_buffer = list(self.detection_buffer[cav_id].keys())
        for frame in frames_in_buffer:
            if frame not in frames_to_keep:
                del self.detection_buffer[cav_id][frame]

    def get_delayed_detection(self, cav_id, current_frame):
        """
        Get detection data from corresponding time based on delay configuration

        Args:
            cav_id: Vehicle ID
            current_frame: Current frame number

        Returns:
            Delayed detection data, or None if not available
        """
        delay = self.delay_config.get(cav_id, 0)
        target_frame = current_frame - delay

        if cav_id not in self.detection_buffer:
            return None

        if target_frame not in self.detection_buffer[cav_id]:
            # If target frame not in buffer, try to return most recent available frame
            available_frames = sorted(self.detection_buffer[cav_id].keys())
            if not available_frames:
                return None

            # Find closest frame not exceeding target_frame
            valid_frames = [f for f in available_frames if f <= target_frame]
            if valid_frames:
                target_frame = max(valid_frames)
                print(f"Warning: Frame {current_frame - delay} not in buffer for {cav_id}, "
                      f"using frame {target_frame} instead")
            else:
                print(f"Warning: No valid delayed frame for {cav_id} at frame {current_frame}")
                return None

        return self.detection_buffer[cav_id][target_frame]

    def get_effective_delay(self, cav_id, current_frame):
        """Get actual delay in frames (considering buffer availability)"""
        delay = self.delay_config.get(cav_id, 0)
        target_frame = current_frame - delay

        if cav_id not in self.detection_buffer:
            return 0

        if target_frame in self.detection_buffer[cav_id]:
            return delay

        available_frames = sorted(self.detection_buffer[cav_id].keys())
        if not available_frames:
            return 0

        valid_frames = [f for f in available_frames if f <= target_frame]
        if valid_frames:
            actual_frame = max(valid_frames)
            return current_frame - actual_frame

        # q_ablation_mode = 'lstm_full'
        #
        # if q_ablation_mode == 'full':
        #     # Full model
        #     self.q_net_dict = q_net_dict
        #     self.q_loss_fn = TrajectoryBasedQLoss()
        #
        # elif q_ablation_mode == 'lstm_diag':
        #     # Simplified version 1: LSTM + diagonal Q
        #     self.q_net_dict = {
        #         'ego': SimplifiedQNet_Diagonal(q_learning_config, device, dtype)
        #     }
        #     self.q_loss_fn = TrajectoryBasedQLoss()
        #
        # elif q_ablation_mode == 'lstm_full':
        #     # Simplified version 2: LSTM + full covariance Q
        #     self.q_net_dict = {
        #         'ego': SimplifiedQNet_FullCov(q_learning_config, device, dtype)
        #     }
        #     self.q_loss_fn = TrajectoryBasedQLoss()
    def update_with_hybrid_ci(self, matched_detections_dict, frame, intermediate_tracks, is_training=False):
        """
        Use hybrid strategy: first perform Kalman filter update for each CAV separately, then fuse using CI algorithm

        Args:
            matched_detections_dict: Dictionary with cav_id as key and matched detection list as value
            frame: Current frame number
            intermediate_tracks: Dictionary with cav_id as key and track copy as value
            is_training: Whether in training mode
        """
        ci = self.covariance_intersection

        # Step 1: Perform independent Kalman filter update for each CAV's track copy
        for cav_id, matched_data in matched_detections_dict.items():
            cav_tracks = intermediate_tracks[cav_id]
            # if cav_id == 'ego':
            # Process each matched pair
            for match in matched_data:
                det_idx, trk_idx, bbox3d, R = match

                # Get corresponding track copy
                trk = cav_tracks[trk_idx]

                # Perform orientation correction
                trk_bbox3d = torch.tensor(bbox3d, dtype=self.dtype, device=self.device)
                trk.dkf.x[3], trk_bbox3d[3] = self.orientation_correction_torch(trk.dkf.x[3], trk_bbox3d[3])

                # Standard Kalman filter update
                trk.dkf.update(trk_bbox3d, R, None)
                trk.time_since_update = 0
                trk.hits += 1
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])

        # Step 2: For each track, collect and fuse update results from different CAVs
        for trk_idx, trk in enumerate(self.trackers):
            # Collect update results from all CAVs that matched this track
            means = []
            covariances = []

            # First add original predicted state
            means.append(trk.dkf.x)
            covariances.append(trk.dkf.P)

            # Collect updated states from each CAV's track copy
            cav_detections = {}  # Record matching information

            for cav_id, cav_tracks in intermediate_tracks.items():
                # Check if this CAV has detections matching this track
                matched = False
                for match in matched_detections_dict.get(cav_id, []):
                    if match[1] == trk_idx:
                        det_idx = match[0]
                        matched = True
                        cav_detections[cav_id] = det_idx
                        break

                # If matched, collect updated track state from this CAV
                if matched:
                    cav_trk = cav_tracks[trk_idx]
                    means.append(cav_trk.dkf.x)
                    covariances.append(cav_trk.dkf.P)

            # If at least one CAV detected this track, use CI fusion
            if len(means) > 1:
                # Use CI algorithm for fusion
                fused_mean, fused_cov = ci.fuse(means, covariances)

                # Update main track state
                trk.dkf.x = fused_mean
                trk.dkf.P = fused_cov
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])

                # Update track statistics
                trk.time_since_update = 0
                trk.hits += 1
                trk.last_updated_frame = frame

                # Update matching information
                for cav_id, det_idx in cav_detections.items():
                    trk.matched_detection_id_dict[cav_id] = det_idx

        # Additional processing in training mode
        if is_training and hasattr(ci, 'get_average_loss'):
            self.ci_stats = {
                'ci_fusion_loss': ci.get_average_loss(),
                'frame': frame
            }

    def update_with_information_form(self, matched_detections_dict, frame):
        """
        Use information form for multi-sensor Kalman filter update
        This method is mathematically optimal for independent measurements
        """
        for trk_idx, trk in enumerate(self.trackers):
            # Collect all detections matching this track
            matched_measurements = []

            for cav_id, matches in matched_detections_dict.items():
                for match in matches:
                    if match[1] == trk_idx:  # Matched to current track
                        det_idx, _, bbox3d, R = match

                        # Orientation correction
                        bbox3d_tensor = torch.tensor(bbox3d, dtype=self.dtype, device=self.device)
                        trk.dkf.x[3], bbox3d_tensor[3] = self.orientation_correction_torch(
                            trk.dkf.x[3], bbox3d_tensor[3])

                        matched_measurements.append({
                            'z': bbox3d_tensor.reshape(-1, 1),  # Observation vector
                            'R': R,  # Observation covariance
                            'cav_id': cav_id,
                            'det_idx': det_idx
                        })

            if not matched_measurements:
                continue  # No matched measurements

            # Convert to information form
            info_matrix = torch.inverse(trk.dkf.P)  # Information matrix
            info_vector = torch.matmul(info_matrix, trk.dkf.x)  # Information vector

            # Observation matrix H (maps state space to observation space)
            H = torch.zeros((7, 10), dtype=self.dtype, device=self.device)
            H[:7, :7] = torch.eye(7, dtype=self.dtype, device=self.device)

            # Accumulate information contribution from each independent measurement
            for measurement in matched_measurements:
                z = measurement['z']
                R = measurement['R']

                # Calculate observation information contribution
                R_inv = torch.inverse(R)
                H_T_R_inv = torch.matmul(H.t(), R_inv)

                # Update information matrix and information vector
                info_matrix += torch.matmul(H_T_R_inv, H)
                info_vector += torch.matmul(H_T_R_inv, z)

                # Record matching information
                trk.matched_detection_id_dict[measurement['cav_id']] = measurement['det_idx']

            # Convert back to covariance form
            trk.dkf.P = torch.inverse(info_matrix)
            trk.dkf.x = torch.matmul(trk.dkf.P, info_vector)

            # Angle normalization
            trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])

            # Update track statistics
            trk.time_since_update = 0
            trk.hits += 1
            trk.last_updated_frame = frame

    def compute_trajectory_q_loss_statistics(self):
        """
        Compute trajectory Q loss statistics, format consistent with other losses
        """
        if not hasattr(self, 'current_q_loss_components'):
            return {
                'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                'count': 0
            }

        if len(self.current_q_loss_components) == 0:
            return {
                'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                'count': 0
            }

        # Collect Q loss from all tracks
        total_q_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        valid_count = 0

        for component in self.current_q_loss_components:
            if component['q_loss'] is not None:
                total_q_loss += component['q_loss']
                valid_count += 1

        return {
            'sum': total_q_loss,
            'count': valid_count
        }

    def update_with_hybrid_ci_update_mix_r_q(self, matched_detections_dict, frame, intermediate_tracks):
        """
        Use hybrid strategy: first perform Kalman filter update for each CAV separately, then fuse using CI algorithm

        Args:
            matched_detections_dict: Dictionary with cav_id as key and matched detection list as value
            frame: Current frame number
            intermediate_tracks: Dictionary with cav_id as key and track copy as value
            is_training: Whether in training mode
        """
        ci = self.covariance_intersection

        # Step 1: Perform independent Kalman filter update for each CAV's track copy
        for cav_id, matched_data in matched_detections_dict.items():
            cav_tracks = intermediate_tracks[cav_id]
            if cav_id == 'ego':
                # Process each matched pair
                for match in matched_data:
                    det_idx, trk_idx, bbox3d, R = match

                    # Get corresponding track copy
                    trk = cav_tracks[trk_idx]

                    # Perform orientation correction
                    trk_bbox3d = torch.tensor(bbox3d, dtype=self.dtype, device=self.device)
                    trk.dkf.x[3], trk_bbox3d[3] = self.orientation_correction_torch(trk.dkf.x[3], trk_bbox3d[3])

                    # Standard Kalman filter update
                    trk.dkf.update(trk_bbox3d, R, None)
                    trk.time_since_update = 0
                    trk.hits += 1
                    trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])
            # elif cav_id == '1':
            #     for match in matched_data:
            #         det_idx, trk_idx, bbox3d, R = match
            #         trk = cav_tracks[trk_idx]
            #         # Perform orientation correction
            #         trk_bbox3d = torch.tensor(bbox3d, dtype=self.dtype, device=self.device)
            #         trk.dkf.x[3], trk_bbox3d[3] = self.orientation_correction_torch(trk.dkf.x[3], trk_bbox3d[3])
            #         # trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])

        # Step 2: For each track, collect and fuse update results from different CAVs
        for trk_idx, trk in enumerate(self.trackers):
            # Collect update results from all CAVs that matched this track
            means = []
            covariances = []

            # First add original predicted state
            means.append(trk.dkf.x)
            covariances.append(trk.dkf.P)

            # Collect updated states from each CAV's track copy
            cav_detections = {}  # Record matching information

            for cav_id, cav_tracks in intermediate_tracks.items():

                # if cav_id == 'ego':
                # Check if this CAV has detections matching this track
                matched = False

                for match in matched_detections_dict.get(cav_id, []):

                    if match[1] == trk_idx:
                        det_idx = match[0]
                        matched = True

                        cav_detections[cav_id] = det_idx
                        break

                # If matched, collect updated track state from this CAV
                if matched:
                    cav_trk = cav_tracks[trk_idx]
                    means.append(cav_trk.dkf.x)
                    covariances.append(cav_trk.dkf.P)

            # If at least one CAV detected this track, use CI fusion
            if len(means) > 1:
                # if len(means) >2:
                #     means = means[1:]
                #     covariances = covariances[1:]
                # Use CI algorithm for fusion
                fused_mean, fused_cov = ci.fuse(means, covariances)

                # Update main track state
                trk.dkf.x = fused_mean
                trk.dkf.P = fused_cov
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])

                # Update track statistics
                trk.time_since_update = 0
                trk.hits += 1
                trk.last_updated_frame = frame

                # Update matching information
                for cav_id, det_idx in cav_detections.items():
                    trk.matched_detection_id_dict[cav_id] = det_idx

    def prediction(self, default_init_Q):
        """
        Override prediction step to use adaptive Q matrix
        """
        self.history_manager.update_histories(self.trackers, self.frame_count)

        trks = []
        q_loss_components = []

        # Current frame statistics
        frame_stats = {
            'frame': self.frame_count,
            'total_tracks': len(self.trackers),
            'adaptive_q_count': 0,
            'default_q_count': 0,
            'track_details': []
        }

        for t in range(len(self.trackers)):
            kf_tmp = self.trackers[t]

            if kf_tmp.id == self.debug_id:
                print('\n before prediction')
                print(kf_tmp.dkf.x.reshape((-1)))
                print('\n current velocity')
                print(kf_tmp.get_velocity())

            # Get state history
            history_data, valid_length = self.history_manager.get_track_history(
                kf_tmp.id, history_length=20
            )

            # Initialize track details (must be defined before use!)
            track_detail = {
                'track_id': kf_tmp.id,
                'valid_length': valid_length,
                'used_adaptive': False
            }

            # Compute adaptive Q matrix
            if valid_length >= 3 and self.enable_learnable_Q:
                try:
                    adaptive_Q, motion_probs, complexity_score = self.q_net_dict['ego'](
                        history_data, default_init_Q
                    )

                    # Statistics
                    self.q_usage_stats['adaptive_q_used'] += 1
                    frame_stats['adaptive_q_count'] += 1
                    track_detail['used_adaptive'] = True
                    track_detail['Q_trace'] = torch.trace(adaptive_Q).item()

                    # Only compute loss during training
                    if self.is_training:
                        q_loss, q_loss_dict = self.q_loss_fn(
                            adaptive_Q, motion_probs, complexity_score, default_init_Q
                        )
                        q_loss_components.append({
                            'track_id': kf_tmp.id,
                            'q_loss': q_loss,
                            'q_loss_dict': q_loss_dict,
                            'adaptive_Q': adaptive_Q
                        })

                    track_specific_Q = adaptive_Q

                except Exception as e:
                    print(f"Error computing Q for track {kf_tmp.id}: {e}")
                    import traceback
                    traceback.print_exc()

                    track_specific_Q = None
                    self.q_usage_stats['default_q_used'] += 1
                    frame_stats['default_q_count'] += 1
                    track_detail['used_adaptive'] = False
                    track_detail['error'] = str(e)
            else:
                track_specific_Q = None
                self.q_usage_stats['default_q_used'] += 1
                frame_stats['default_q_count'] += 1
                track_detail['Q_trace'] = torch.trace(default_init_Q).item()

            # Record statistics
            self.q_usage_stats['total_predictions'] += 1
            frame_stats['track_details'].append(track_detail)

            # Kalman filter prediction
            kf_tmp.dkf.predict(track_specific_Q)

            if kf_tmp.id == self.debug_id:
                print('After prediction')
                print(kf_tmp.dkf.x.reshape((-1)))

            kf_tmp.dkf.x[3] = self.within_range_torch(kf_tmp.dkf.x[3])

            # for visualization
            kf_tmp.reset_matched_detection_id_dict()

            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.dkf.x.reshape((-1))[:7]
            trk_tmp = trk_tmp.detach().cpu().numpy()
            trks.append(Box3D.array2bbox(trk_tmp))

        self.q_usage_stats['frames_processed'] += 1
        self.q_usage_stats['per_frame_stats'].append(frame_stats)

        # Print summary every 50 frames
        if self.frame_count > 0 and self.frame_count % 50 == 0:
            total = self.q_usage_stats['total_predictions']
            adaptive = self.q_usage_stats['adaptive_q_used']
            default = self.q_usage_stats['default_q_used']

            if total > 0:
                print(f"\n{'=' * 60}")
                print(f"Q Usage Statistics (up to frame {self.frame_count}):")
                print(f"  Total predictions: {total}")
                print(f"  Adaptive Q used: {adaptive} ({adaptive / total * 100:.1f}%)")
                print(f"  Default Q used: {default} ({default / total * 100:.1f}%)")
                print(f"  Current frame tracks: {frame_stats['total_tracks']}")
                print(f"    - Using adaptive Q: {frame_stats['adaptive_q_count']}")
                print(f"    - Using default Q: {frame_stats['default_q_count']}")
                print('=' * 60)

        self.current_q_loss_components = q_loss_components
        return trks

    def get_final_q_statistics(self):
        """Called at end of sequence to get final statistics"""
        total = self.q_usage_stats['total_predictions']
        if total == 0:
            return None

        stats = {
            'total_predictions': total,
            'adaptive_q_used': self.q_usage_stats['adaptive_q_used'],
            'default_q_used': self.q_usage_stats['default_q_used'],
            'adaptive_q_percentage': self.q_usage_stats['adaptive_q_used'] / total * 100,
            'frames_processed': self.q_usage_stats['frames_processed']
        }
        return stats

    def get_empirical_covariance(self, track_id, innovation_histories):
        """Progressive computation: start when enough samples available, quality improves with more samples"""
        if track_id not in innovation_histories:
            return None, None

        innovations = list(innovation_histories[track_id])

        # Minimum 5 samples needed, but no need to wait for 20
        if len(innovations) < self.min_samples:
            return None, None

        # Use all currently available samples (between 5 and 20)
        innovation_matrix = torch.stack(innovations)
        S_empirical = torch.cov(innovation_matrix.T)

        # Return covariance matrix and confidence weight
        confidence = min(len(innovations) / self.window_size, 1.0)
        return S_empirical, confidence

    def compute_joint_loss(self, matched, unmatched_trks, dets):
        s_empirical_list = []
        confidence_list = []
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                if len(d) == 1:
                    # Compute innovation vector
                    bbox3d = Box3D.bbox2array(dets[d[0]])
                    predicted_bbox = trk.dkf.x[:7, 0].detach().clone()
                    innovation = torch.tensor(bbox3d, dtype=self.dtype, device=self.device) - predicted_bbox

                    # Update innovation sequence history
                    innovation_histories = self.q_manager.update_innovation_history(trk.id, innovation)
                    s_empirical, confidence = self.get_empirical_covariance(trk.id, innovation_histories)
                    s_empirical_list.append(s_empirical)
                    confidence_list.append(confidence)
                    # Empirical covariance calculation
                    if trk.id not in self.s_empirical:
                        if s_empirical is not None:
                            self.s_empirical[trk.id] = {
                                's_empirical': s_empirical_list,  # State vector history
                                'covariances': confidence_list}  # Covariance matrix history

        return self.s_theoretical, self.s_empirical

    def compute_joint_rq_loss_unified(self, matched, unmatched_trks, dets, learnable_R_dict):
        """
        Unified RQ loss computation, ensuring s_theoretical and s_empirical are synchronized
        """
        total_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        valid_pairs = 0

        for t, trk in enumerate(self.trackers):
            if t in unmatched_trks:  # Skip unmatched tracks
                continue
            kf_tmp = self.trackers[t]
            track_id = trk.id
            state_history, state_valid_length = self.history_manager.get_track_history(
                track_id, history_length=20
            )
            d = matched[np.where(matched[:, 1] == t)[0], 0]
            # if len(d) == 1 and state_valid_length >= 5:
            if len(d) == 1:
                # Compute innovation vector
                bbox3d = Box3D.bbox2array(dets[d[0]])
                predicted_bbox = trk.dkf.x[:7, 0].detach().clone()
                innovation = torch.tensor(bbox3d, dtype=self.dtype, device=self.device) - predicted_bbox
                # innovation_history = self.q_manager.innovation_histories.get(track_id, [])
                innovation_histories = self.q_manager.update_innovation_history(trk.id, innovation)
                self.innovations_id_list = list(innovation_histories[track_id])
            # 1. Unified condition check: need both sufficient state history and innovation history

            # innovation_history = self.q_manager.innovation_histories.get(track_id, [])

            # Key: only proceed with computation when both conditions are met
            if len(self.innovations_id_list) >= self.min_samples:

                # 2. Compute s_theoretical (based on current state)
                try:
                    # Compute theoretical S
                    s_theoretical = kf_tmp.dkf.compute_S_theoretical_with_predicted_P(learnable_R_dict[d[0]])

                except Exception as e:
                    print(f"Error computing s_theoretical for track {track_id}: {e}")
                    continue

                # 3. Compute s_empirical (based on innovation history)
                try:
                    innovation_matrix = torch.stack(self.innovations_id_list)
                    s_empirical = torch.cov(innovation_matrix.T)
                    confidence = min(len(self.innovations_id_list) / self.window_size, 1.0)
                    if s_empirical is None:
                        continue

                except Exception as e:
                    print(f"Error computing s_empirical for track {track_id}: {e}")
                    continue

                # 4. Compute loss
                diff = s_theoretical - s_empirical
                base_loss = F.mse_loss(s_theoretical, s_empirical)

                weighted_loss = confidence * base_loss  # Weight by number of samples

                total_loss += weighted_loss
                valid_pairs += 1

                # 5. Optional: store for debugging
                self.s_theoretical[track_id] = s_theoretical
                self.s_empirical[track_id] = s_empirical

        return total_loss / max(valid_pairs, 1), valid_pairs

    def kalman_update(self, prior_x, prior_P, bbox3d, R):
        """
        Independent Kalman filter update function for CI fusion

        Args:
            prior_x: Prior state vector [10, 1]
            prior_P: Prior state covariance matrix [10, 10]
            bbox3d: Observation vector [7] - [x, y, z, theta, l, w, h]
            R: Observation covariance matrix [7, 7]

        Returns:
            updated_x: Updated state vector [10, 1]
            updated_P: Updated state covariance matrix [10, 10]
        """
        # Ensure input is in correct tensor format
        if not isinstance(bbox3d, torch.Tensor):
            bbox3d = torch.tensor(bbox3d, dtype=self.dtype, device=self.device)

        # Perform orientation correction (handle angle periodicity)
        corrected_prior_x = prior_x.clone()
        corrected_prior_x[3], bbox3d[3] = self.orientation_correction_torch(
            prior_x[3], bbox3d[3]
        )

        # Observation matrix H: maps 10-dim state to 7-dim observation
        # H = [I_7x7, 0_7x3] - only observe position, angle, size, not velocity
        H = torch.zeros((7, 10), dtype=self.dtype, device=self.device)
        H[:7, :7] = torch.eye(7, dtype=self.dtype, device=self.device)

        # Kalman filter update steps

        # 1. Compute predicted observation
        z_pred = torch.matmul(H, corrected_prior_x)  # [7, 1]

        # 2. Compute innovation vector (residual)
        z_obs = bbox3d.reshape(-1, 1)  # [7, 1]
        innovation = z_obs - z_pred  # [7, 1]

        # Handle angle innovation periodicity
        innovation[3] = self.within_range_torch(innovation[3])

        # 3. Compute innovation covariance matrix
        S = torch.matmul(torch.matmul(H, prior_P), H.t()) + R  # [7, 7]

        # 4. Compute Kalman gain
        try:
            S_inv = torch.inverse(S)
        except:
            # Use pseudo-inverse if S is not invertible
            S_inv = torch.pinverse(S)

        K = torch.matmul(torch.matmul(prior_P, H.t()), S_inv)  # [10, 7]

        # 5. Update state estimate
        updated_x = corrected_prior_x + torch.matmul(K, innovation)  # [10, 1]

        # 6. Update state covariance (Joseph form, numerically stable)
        I_KH = torch.eye(10, dtype=self.dtype, device=self.device) - torch.matmul(K, H)
        updated_P = torch.matmul(torch.matmul(I_KH, prior_P), I_KH.t()) + \
                    torch.matmul(torch.matmul(K, R), K.t())  # [10, 10]

        # 7. Angle normalization
        updated_x[3] = self.within_range_torch(updated_x[3])

        return updated_x, updated_P

    def track_multi_sensor_differentiable_kalman_filter(self, dets_all_dict, frame, seq_name, cav_id_list,
                                                        dets_feature_dict, gt_boxes, gt_ids,
                                                        transformation_matrix_dict):
        """
        Multi-sensor cooperative tracking algorithm preserving sequential processing advantages - supports delay simulation
        """
        measure_run_time = False
        loss_dict = {}

        # Step 1: Add current frame detection data to buffer
        for cav_id in dets_all_dict.keys():
            dets = dets_all_dict[cav_id]['dets']
            info = dets_all_dict[cav_id]['info']
            feature = dets_feature_dict[cav_id]
            transformation_matrix = transformation_matrix_dict[cav_id]

            # Add to buffer
            self.add_detection_to_buffer(
                cav_id, frame, dets, info, feature, transformation_matrix
            )

        # Step 2: Get delayed detection data from buffer
        dets_dict = {}
        info_dict = {}
        dets_feature_delayed = {}
        transformation_matrix_delayed = {}
        actual_delays = {}  # Record actually used delays

        # Use delayed data
        for cav_id in cav_id_list:
            delayed_data = self.get_delayed_detection(cav_id, frame)

            if delayed_data is None:
                # No available delayed data, skip this CAV
                print(f"Warning: No delayed detection available for {cav_id} at frame {frame}")
                continue

            dets_dict[cav_id] = delayed_data['dets']
            info_dict[cav_id] = delayed_data['info']
            dets_feature_delayed[cav_id] = delayed_data['feature']
            transformation_matrix_delayed[cav_id] = delayed_data['transformation_matrix']

            # Record actual delay
            actual_delays[cav_id] = self.get_effective_delay(cav_id, frame)

        # Print delay info every 10 frames
        if frame % 10 == 0:
            delay_info = ', '.join([f"{cav}:{actual_delays.get(cav, 0)}"
                                    for cav in cav_id_list])
            print(f"Frame {frame:3d} - Actual delays: {delay_info}")

        # Step 3: Logging and preprocessing
        if self.debug_id:
            print('\nframe is %s' % frame)

        print_str = f'\n\n*****************************************\n\n' \
                    f'processing seq_name/frame {seq_name}/{frame}'
        print_log(print_str, log=self.log, display=False)
        self.frame_count += 1

        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        # Convert detection data to ground truth order
        dets_in_gt_order_dict = {}
        for cav_id in dets_dict.keys():  # Use delayed data
            dets_in_gt_order_dict[cav_id] = self.process_dets_to_gt_order(dets_dict[cav_id])

        # Convert detection data to Box3D format
        for cav_id in dets_dict.keys():  # Use delayed data
            dets_dict[cav_id] = self.process_dets(dets_dict[cav_id])

        # Step 4: Covariance estimation (using delayed data)
        default_init_P, default_init_Q, _ = DKF.get_ab3dmot_default_covariance_matrices(
            self.dtype, self.device, dim_x=10, dim_z=7)

        observation_covariance_dict = {}
        learnable_init_P_dict = {}
        learnable_R_dict = {}
        det_neg_log_likelihood_loss_dict = {}
        det_neg_log_likelihood_loss_sum = []
        det_neg_log_likelihood_loss_count = 0

        for cav_id in dets_dict.keys():  # Use delayed data
            # Use delayed feature data
            dets_feature = dets_feature_delayed[cav_id]
            dets_feature = torch.tensor(dets_feature, dtype=self.dtype, device=self.device)

            transformation_matrix = transformation_matrix_delayed[cav_id]
            transformation_matrix = torch.tensor(
                transformation_matrix, dtype=self.dtype, device=self.device)

            dets_in_gt_order = dets_in_gt_order_dict[cav_id]
            dets_in_gt_order = torch.tensor(
                dets_in_gt_order, dtype=self.dtype, device=self.device)

            # Use covariance network
            if self.use_multiple_nets:
                observation_covariance_dict[cav_id] = \
                    self.observation_covariance_net_dict[cav_id](
                        dets_feature, frame, transformation_matrix, dets_in_gt_order)
            else:
                observation_covariance_dict[cav_id] = \
                    self.observation_covariance_net_dict['ego'](
                        dets_feature, frame, transformation_matrix, dets_in_gt_order)

            # Generate learnable covariance matrices
            learnable_init_P_dict[cav_id], learnable_R_dict[cav_id] = \
                self.get_learnable_observation_covariance(
                    default_init_P, observation_covariance_dict[cav_id])

            # Compute negative log-likelihood loss
            if not measure_run_time:
                if dets_in_gt_order.shape[0] == 0:
                    continue
                det_neg_log_likelihood_loss_dict[cav_id], matched_det_count = \
                    get_neg_log_likelihood_loss(
                        dets_in_gt_order, learnable_R_dict[cav_id], gt_boxes)
                det_neg_log_likelihood_loss_sum.append(
                    det_neg_log_likelihood_loss_dict[cav_id])
                det_neg_log_likelihood_loss_count += matched_det_count

        # Aggregate negative log-likelihood loss
        if measure_run_time:
            loss_dict['det_neg_log_likelihood'] = {
                'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                'count': 1
            }
        else:
            if det_neg_log_likelihood_loss_sum:
                det_neg_log_likelihood_loss_sum = torch.cat(
                    det_neg_log_likelihood_loss_sum, dim=0)
                det_neg_log_likelihood_loss_sum = torch.sum(
                    det_neg_log_likelihood_loss_sum)
            else:
                det_neg_log_likelihood_loss_sum = torch.zeros(
                    1, dtype=self.dtype, device=self.device)

            loss_dict['det_neg_log_likelihood'] = {
                'sum': det_neg_log_likelihood_loss_sum,
                'count': det_neg_log_likelihood_loss_count
            }

        # Subsequent steps remain the same
        # KF prediction, association loss, CI fusion, etc. are exactly the same as before

        if self.force_gt_as_predicted_track and self.prev_prev_gt_boxes is not None:
            trks = self.transform_gt_as_predicted_track(frame)
        else:
            trks = self.prediction(default_init_Q)

        # Q_net loss statistics
        if self.enable_learnable_Q and self.is_training:
            trajectory_q_loss_stats = self.compute_trajectory_q_loss_statistics()
            loss_dict['trajectory_q_loss'] = trajectory_q_loss_stats
        else:
            loss_dict['trajectory_q_loss'] = {
                'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                'count': 1
            }

        # ego motion compensation
        if (frame > 0) and (self.ego_com) and (self.oxts is not None):
            trks = self.ego_motion_compensation(frame, trks)

        # Association loss
        if measure_run_time:
            loss_dict['association'] = {
                'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                'count': 1
            }
        else:
            association_loss_sum, association_loss_count = get_association_loss(
                dets_dict, self.trackers, gt_boxes, gt_ids,
                self.prev_gt_boxes, self.prev_gt_ids)
            loss_dict['association'] = {
                'sum': association_loss_sum,
                'count': association_loss_count
            }

        # Sequential processing with CI fusion
        track_fusion_buffer = {}
        for track_idx in range(len(self.trackers)):
            track_fusion_buffer[track_idx] = {
                'original_prior_state': (
                    self.trackers[track_idx].dkf.x.clone(),
                    self.trackers[track_idx].dkf.P.clone()),
                'cav_updates': {},
                'cav_list': []
            }

        # Process each CAV sequentially
        for cav_id in cav_id_list:
            if cav_id not in dets_dict.keys():
                continue

            # Data association
            trk_innovation_matrix = None
            if self.metric == 'm_dis':
                trk_innovation_matrix = [
                    trk.compute_innovation_matrix().detach().cpu().numpy()
                    for trk in self.trackers
                ]

            matched, unmatched_dets, unmatched_trks, cost, affi = \
                data_association(dets_dict[cav_id], trks, self.metric,
                                 self.thres, self.algm, trk_innovation_matrix)

            self.process_matched_tracks_with_ci_buffer(
                matched, unmatched_trks, dets_dict[cav_id], info_dict[cav_id],
                learnable_R_dict, frame, cav_id, track_fusion_buffer
            )

            # S loss computation
            if self.enable_learnable_Q and self.is_training:
                s_loss, valid_pairs = self.compute_joint_rq_loss_unified(
                    matched, unmatched_trks, dets_dict[cav_id],
                    learnable_R_dict[cav_id])
                loss_dict['trajectory_s_loss'] = {
                    'sum': s_loss,
                    'count': max(valid_pairs, 1)
                }
            else:
                loss_dict['trajectory_s_loss'] = {
                    'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                    'count': 1
                }

            # Process unmatched detections
            new_id_list = self.birth(dets_dict[cav_id], info_dict[cav_id],
                                     unmatched_dets, frame, cav_id,
                                     learnable_init_P_dict)

            for new_track_idx in range(len(self.trackers) - len(new_id_list),
                                       len(self.trackers)):
                track_fusion_buffer[new_track_idx] = {
                    'original_prior_state': (
                        self.trackers[new_track_idx].dkf.x.clone(),
                        self.trackers[new_track_idx].dkf.P.clone()),
                    'cav_updates': {},
                    'cav_list': []
                }

            # Update trks
            trks = self.get_trks_for_match()

        # Apply CI fusion
        self.apply_ci_fusion_from_buffer(track_fusion_buffer, frame,
                                         is_training=not measure_run_time)

        # Regression loss
        if measure_run_time:
            loss_dict['regression'] = {
                'sum': torch.zeros(1, dtype=self.dtype, device=self.device),
                'count': 1
            }
        else:
            regression_loss_sum, regression_loss_count = self.get_regression_loss(gt_boxes)
            loss_dict['regression'] = {
                'sum': regression_loss_sum,
                'count': regression_loss_count
            }

        # Save GT information
        self.prev_prev_gt_boxes = self.prev_gt_boxes
        self.prev_prev_gt_ids = self.prev_gt_ids
        self.prev_gt_boxes = gt_boxes
        self.prev_gt_ids = gt_ids

        # Output results
        results, matched_detection_id_dict, track_P = self.output()

        if len(results) > 0:
            results = [np.concatenate(results)]
        else:
            results = [np.empty((0, 15))]

        self.id_now_output = results[0][:, 7].tolist()

        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)

        return results, affi, loss_dict, matched_detection_id_dict, \
            learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict

    def process_matched_tracks_with_ci_buffer(self, matched, unmatched_trks, dets, info, learnable_R_dict, frame,
                                              cav_id, track_fusion_buffer):
        """
        Process matched tracks, intelligently decide whether to update directly or prepare for CI fusion
        """
        assert (len(dets) == learnable_R_dict[cav_id].shape[0])

        dets = copy.copy(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                assert len(d) == 1, 'error'
                det_idx = d[0]

                # Prepare detection data
                bbox3d = Box3D.bbox2array(dets[det_idx])
                learnable_R = learnable_R_dict[cav_id][det_idx]

                # Add current CAV to detection list
                track_fusion_buffer[t]['cav_list'].append(cav_id)

                # Compute update result based on original prior state
                original_prior_x, original_prior_P = track_fusion_buffer[t]['original_prior_state']
                updated_x, updated_P = self.perform_kalman_update_on_state(
                    original_prior_x.clone(), original_prior_P.clone(), bbox3d, learnable_R
                )

                # Record update result to buffer
                track_fusion_buffer[t]['cav_updates'][cav_id] = {
                    'updated_x': updated_x,
                    'updated_P': updated_P,
                    'det_idx': det_idx,
                    'bbox3d': bbox3d,
                    'learnable_R': learnable_R,
                    'info': info[det_idx]
                }

                # Decision: direct update vs prepare for CI fusion
                # if len(track_fusion_buffer[t]['cav_list']) == 1:
                # This is first CAV detecting this track, update main track directly (preserve sequential processing advantage)
                trk.dkf.x[3], bbox3d[3] = self.orientation_correction_torch(trk.dkf.x[3], bbox3d[3])

                trk.dkf.update(bbox3d, learnable_R, None)
                trk.time_since_update = 0
                trk.hits += 1
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])
                trk.last_updated_frame = frame
                trk.last_updated_cav_id = cav_id
                trk.matched_detection_id_dict[cav_id] = det_idx
                trk.info = info[det_idx]

                # 2nd and subsequent CAVs: only record, don't fuse immediately, wait for final unified processing

    def perform_kalman_update_on_state(self, x, P, bbox3d, R):
        """
        Perform Kalman filter update on given state without modifying original state
        """
        # Perform orientation correction
        bbox3d = torch.tensor(bbox3d, dtype=self.dtype, device=self.device)
        x[3], bbox3d[3] = self.orientation_correction_torch(x[3], bbox3d[3])

        # Observation matrix H (7x10)
        H = torch.zeros((7, 10), dtype=self.dtype, device=self.device)
        H[:7, :7] = torch.eye(7, dtype=self.dtype, device=self.device)

        # Compute Kalman gain
        S = torch.matmul(torch.matmul(H, P), H.t()) + R  # Innovation covariance
        K = torch.matmul(torch.matmul(P, H.t()), torch.inverse(S))  # Kalman gain

        # Update state and covariance
        z = bbox3d.reshape(-1, 1)
        y = z - torch.matmul(H, x)  # Innovation
        x_updated = x + torch.matmul(K, y)
        P_updated = P - torch.matmul(torch.matmul(K, H), P)

        x_updated[3] = self.within_range_torch(x_updated[3])

        return x_updated, P_updated

    def apply_ci_fusion_from_buffer(self, track_fusion_buffer, frame, is_training=False):
        """
        Apply CI fusion to tracks detected by 2 or more CAVs (one-time processing)
        """
        ci = self.covariance_intersection
        # print('Normal fusion')
        # means = []
        # covariances = []
        # for trk_idx, trk in enumerate(self.trackers):
        #     # Collect all update results from CAVs that matched this track
        #
        #
        #     # First add original predicted state
        #     means.append(trk.dkf.x)
        #     covariances.append(trk.dkf.P)
        for track_idx, buffer_data in track_fusion_buffer.items():
            # if len(buffer_data['cav_list']) >= 2:  # 2 or more CAVs
            # Collect all update results
            means = []
            covariances = []
            main_trk = self.trackers[track_idx]
            # means.append(main_trk.dkf.x)
            # covariances.append(main_trk.dkf.P)
            # Add original prior state
            # original_prior_x, original_prior_P = buffer_data['original_prior_state']
            # means.append(original_prior_x)
            # covariances.append(original_prior_P)

            # Add update results from each CAV
            cav_detections = {}
            for cav_id in buffer_data['cav_list']:
                update_data = buffer_data['cav_updates'][cav_id]
                means.append(update_data['updated_x'])
                covariances.append(update_data['updated_P'])
                cav_detections[cav_id] = update_data['det_idx']

            # Apply CI fusion
            if len(means) > 1:
                fused_mean, fused_cov = ci.fuse(means, covariances)
                # Update main track
                trk = self.trackers[track_idx]
                trk.dkf.x = fused_mean
                trk.dkf.P = fused_cov
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])

                # Update track statistics
                trk.time_since_update = 0
                trk.hits += 1
                trk.last_updated_frame = frame

                # Update matching information - safely get last CAV ID
                if buffer_data['cav_list']:  # Ensure list is not empty
                    last_cav_id = buffer_data['cav_list'][-1]
                    if last_cav_id in buffer_data['cav_updates']:
                        trk.info = buffer_data['cav_updates'][last_cav_id]['info']

                # Record all matched detections
                trk.matched_detection_id_dict = {}  # Reset
                for cav_id, det_idx in cav_detections.items():
                    trk.matched_detection_id_dict[cav_id] = det_idx
            elif len(means) == 1:
                trk = self.trackers[track_idx]
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])
                # Update track statistics
                trk.time_since_update = 0
                trk.hits += 1
                trk.last_updated_frame = frame

                # Update matching information - safely get last CAV ID
                if buffer_data['cav_list']:  # Ensure list is not empty
                    last_cav_id = buffer_data['cav_list'][-1]
                    if last_cav_id in buffer_data['cav_updates']:
                        trk.info = buffer_data['cav_updates'][last_cav_id]['info']

                # Record all matched detections
                trk.matched_detection_id_dict = {}  # Reset
                for cav_id, det_idx in cav_detections.items():
                    trk.matched_detection_id_dict[cav_id] = det_idx
