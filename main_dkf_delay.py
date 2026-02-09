#
#
#
# from __future__ import print_function
# import matplotlib; matplotlib.use('Agg')
# import os, numpy as np, time, sys, argparse
# #sys.path.append('/ocean/projects/cis230055p/hchiu1/my_cooperative_tracking/AB3DMOT')
# #sys.path.append('/ocean/projects/cis230055p/hchiu1/my_cooperative_tracking/AB3DMOT/Xinshuo_PyToolbox')
# #sys.path.append('/ocean/projects/cis230055p/hchiu1/my_cooperative_tracking/V2V4Real')
# from AB3DMOT.AB3DMOT_libs.utils import Config, get_subfolder_seq
# from AB3DMOT.AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_results, save_affinity
# from AB3DMOT.scripts.KITTI.evaluate import trackingEvaluation, evaluate
# from AB3DMOT.scripts.post_processing.combine_trk_cat import combine_trk_cat
# from AB3DMOT.scripts.KITTI import mailpy
# from AB3DMOT.scripts.KITTI import munkres
# from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing, save_txt_file
# from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import get_timestring, print_log
#
# import torch
# from DMSTrack.model import ObservationCovarianceNet, DMSTrack
# from YJQTrack.q_noise_enhanced.enhanced_dmstrack import EnhancedDMSTrack
# # from YJQTrack.q_noise_enhanced.best_for_now_with_r_q import BestForNowTrack
# # from YJQTrack.q_noise_enhanced.best_for_now import BestForNowTrackV1
# # from YJQTrack.q_noise_enhanced.model_online_correlation import BestForNowTrack
# from YJQTrack.q_noise_enhanced.parallel_enhanced_dmstrack import ParallelEnhancedDMSTrack
# from YJQTrack.visualization import visualize
# from tensorboardX import SummaryWriter
# from YJQTrack.CI.q_noise_net import ProcessNoiseNet
# from  YJQTrack.q_noise_enhanced.q_noise_with_innovation import InnovationBasedQNet
# from  YJQTrack.q_noise_enhanced.q_noise_with_trajectory import TrajectoryBasedQNet
# import os
#
# from YJQTrack.CI.covariance_intersection_deep import CovarianceIntersection
# from viztracer import VizTracer
# from YJQTrack.q_noise_enhanced.serial_dmstrack_with_ci import SeriallDMSTrackWithCI
# # from YJQTrack.q_noise_enhanced.best_for_now import BestForNowTrackV1
# # from YJQTrack.q_noise_enhanced.model_online_correlation import BestForNowTrack
# from YJQTrack.q_noise_enhanced.best_for_now_with_r_q import BestForNowTrack
# from viztracer import VizTracer
# from YJQTrack.q_noise_enhanced.simple_q_for_ral_r1.simple_q_baselines import create_q_net
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='DMSTrack')
#     parser.add_argument('--dataset', type=str, default='v2v4real', help='v2v4real')
#     parser.add_argument('--split', type=str, default='', help='train, val, test')
#     parser.add_argument('--det_name', type=str, default='', help='multi_sensor_differentiable_kalman_filter')
#     parser.add_argument('--seq_eval_mode', type=str, default='all', help='which seq to eval: all, 0000, 0001, ...')
#
#     parser.add_argument('--run_evaluation_every_epoch', action='store_true', help='auto run evaluation every epoch')
#     parser.add_argument('--force_gt_as_predicted_track', action='store_true', help='during training force gt as predicted track before data association and update')
#     parser.add_argument('--use_static_default_R', action='store_true', help='use static default observation noise covariance')
#     parser.add_argument('--use_multiple_nets', action='store_true', help='use multiple observation covariance nets for multiple cars')
#     parser.add_argument('--num_frames_backprop', type=int, default=-1, help='number of past frames that the loss backpropagates. If -1, loss is calculated for the full (sub) sequence.')
#     parser.add_argument('--num_frames_per_sub_seq', type=int, default=-1, help='number of frames per sub seq during training data chunk and random shuffle. If -1, each sequence is a sub sequence.')
#     parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
#     parser.add_argument('--load_model_path', type=str, default='', help='path to ego model checkpoint')
#     parser.add_argument('--save_dir_prefix', type=str, default='', help='prefix of save dir')
#     parser.add_argument('--feature', type=str, default='fusion', help='model feature, fusion or bev or pos')
#     parser.add_argument('--regression_loss_weight', type=float, default=0, help='regression loss weight')
#     parser.add_argument('--association_loss_weight', type=float, default=0, help='association loss weight')
#     parser.add_argument('--det_neg_log_likelihood_loss_weight', type=float, default=0, help='detection negative log likelihood loss weight')
#     parser.add_argument('--clip_grad_norm', type=float, default=-1, help='clip gradient norm up to this value. -1 means no clip')
#     parser.add_argument('--use_reversed_cav_id_list', action='store_true', help='process cav_1 then ego during kf update')
#
#
#     parser.add_argument('--training_split', type=str, default='train', help='data split during training')
#     parser.add_argument('--evaluation_split', type=str, default='val', help='data split during evaluation')
#
#
#     # visualization
#     parser.add_argument('--show_vis', action='store_true',
#                         help='show open3d interactive visualization result')
#     parser.add_argument('--save_vis', action='store_true',
#                         help='save open3d image and video visualization result')
#
#     args = parser.parse_args()
#     return args
#
#
# def get_global_timestamp_index(seq_name, frame, len_record):
#   # only for v2v4real no_fusion_keep_all
#   seq_id = int(seq_name)
#   #print('seq_id: ', seq_id)
#   #print('frame: ', frame)
#   #print('len_record: ', len_record)
#   # test set: [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
#   #print('detection_feature_path: ', detection_feature_path)
#   # /home/eddy/my_cooperative_tracking/V2V4Real/official_models/no_fusion_keep_all/npy/
#
#   if seq_id == 0:
#     start_global_timestamp_index = 0
#   else:
#     start_global_timestamp_index = len_record[seq_id - 1]
#
#   global_timestamp_index = start_global_timestamp_index + frame
#
#   return global_timestamp_index
#
#
# def load_transformation_matrix(global_timestamp_index, detection_feature_path, cav_id):
#   transformation_matrix_file = os.path.join(detection_feature_path, cav_id, '%04d_transformation_matrix.npy' % global_timestamp_index)
#   transformation_matrix = np.load(transformation_matrix_file)
#   #print('transformation_matrix.shape: ', transformation_matrix.shape)
#   # (4, 4)
#   #assert False
#   return transformation_matrix
#
#
#
# def load_detection_feature_cobevt_single_sensor(global_timestamp_index, detection_feature_path, cav_id):
#   detection_feature_file = os.path.join(detection_feature_path, cav_id, '%04d_feature.npy' % global_timestamp_index)
#   #print('detection_feature_file: ', detection_feature_file)
#   detection_feature = np.load(detection_feature_file)
#   #print('detection_feature.shape: ', detection_feature.shape)
#   # (5, 256, 5, 5)
#   #print('np.sum(detection_feature[0], axis=0): ', np.sum(detection_feature[0], axis=0))
#
#   early_detection_feature_file = os.path.join(detection_feature_path, cav_id, '%04d_early_feature.npy' % global_timestamp_index)
#   early_detection_feature = np.load(early_detection_feature_file)
#   #print('early_detection_feature.shape: ', early_detection_feature.shape)
#   # (5, 256, 5, 5)
#
#   # concate the two feature
#   aggregated_detection_feature = np.concatenate([detection_feature, early_detection_feature], axis=1)
#   #print('aggregated_detection_feature.shape: ', aggregated_detection_feature.shape)
#   # (5, 512, 5, 5)
#
#
#   return aggregated_detection_feature
#
#
#
#
# def load_detection_feature(global_timestamp_index, detection_feature_path, cav_id):
#   detection_feature_file = os.path.join(detection_feature_path, cav_id, '%04d_feature.npy' % global_timestamp_index)
#   #print('detection_feature_file: ', detection_feature_file)
#   detection_feature = np.load(detection_feature_file)
#   #print('detection_feature.shape: ', detection_feature.shape)
#   # (5, 256, 5, 5)
#   #print('np.sum(detection_feature[0], axis=0): ', np.sum(detection_feature[0], axis=0))
#
#   # repeat (1, 1, 4, 4)
#   detection_feature = np.repeat(detection_feature, 4, axis=2)
#   detection_feature = np.repeat(detection_feature, 4, axis=3)
#   #print('detection_feature.shape: ', detection_feature.shape)
#   # (5, 256, 20, 20)
#   #print('np.sum(detection_feature[0], axis=0): ', np.sum(detection_feature[0], axis=0))
#
#   early_detection_feature_file = os.path.join(detection_feature_path, cav_id, '%04d_early_feature.npy' % global_timestamp_index)
#   early_detection_feature = np.load(early_detection_feature_file)
#   #print('early_detection_feature.shape: ', early_detection_feature.shape)
#   # (5, 64, 20, 20)
#
#   # concate the two feature
#   aggregated_detection_feature = np.concatenate([detection_feature, early_detection_feature], axis=1)
#   #print('aggregated_detection_feature.shape: ', aggregated_detection_feature.shape)
#   # (5, 320, 20, 20)
#
#   return aggregated_detection_feature
#
#
# def load_ground_truth_data_all(result_sha, seq_eval_mode, split):
#   e = trackingEvaluation(result_sha, evaluate_v2v4real=True, seq_eval_mode=seq_eval_mode, v2v4real_split=split)
#   e.loadGroundtruth()
#
#   #ground_truth = e.groundtruth
#   #print('e.n_gt_seq: ', e.n_gt_seq) # [6, 12, 7, 73, 65, 137, 56, 32, 27]
#   #print('e.n_gt_trajectories: ', e.n_gt_trajectories) # 415
#   #print('len(ground_truth): ', len(ground_truth)) # 9 sequences
#   #print('len(ground_truth[0]): ', len(ground_truth[0])) # 147 frames in the first sequence
#   #print('len(ground_truth[0][0]): ', len(ground_truth[0][0])) # 1
#   #print('ground_truth[0][0][0]: ', ground_truth[0][0][0])
#   #print('ground_truth[0][1][0]: ', ground_truth[0][1][0])
#
#   ground_truth_data_all = e.groundtruth
#   return ground_truth_data_all
#
#
# def get_ground_truth_data_single_frame(seq_idx_in_eval, frame, ground_truth_data_all, dtype, device):
#   '''
#   Input:
#     ground_truth_data_all:
#       list of seq
#         list of frame
#           list of object
#             object: class tData in AB3DMOT/scripts/KITTI/evaluate.py
#   Output:
#     N: number of ground-truth objects in this frame
#     gt_boxes: tensor (N, 7):
#       same order of kalman filter state: [x, y, z, theta, l, w, h]
#     gt_ids: tensor (N)
#   '''
#   ground_truth_data_single_frame = ground_truth_data_all[seq_idx_in_eval][frame]
#   gt_boxes = []
#   gt_ids = []
#
#   for i in range(len(ground_truth_data_single_frame)):
#     object = ground_truth_data_single_frame[i]
#     #print(object)
#     gt_boxes.append(
#       np.array(
#         [object.x, object.y, object.z, object.ry, object.l, object.w, object.h]
#       )
#     )
#     gt_ids.append(object.track_id)
#
#   gt_boxes = np.stack(gt_boxes)
#   gt_ids = np.stack(gt_ids)
#
#   gt_boxes = torch.tensor(gt_boxes, dtype=dtype, device=device)
#   gt_ids = torch.tensor(gt_ids, dtype=dtype, device=device)
#
#   #print('gt_boxes: ', gt_boxes)
#   #print('gt_ids: ', gt_ids)
#
#   #print('gt_boxes.shape: ', gt_boxes.shape)
#   #print('gt_ids.shape: ', gt_ids.shape)
#
#   return gt_boxes, gt_ids
#
#
# # Similar to AB3DMOT_libs/utils.py initialize()
# def initialize_DMSTrack(cfg, seq_name, cat, ID_start, hw, log_file, dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, split,
#                         q_learning_config, enable_learnable_Q=False, q_net_dict=None):
#   # initialize the tracker and provide all path of data needed
#   # 历史管理器配置
#   history_manager_config = {
#     'max_history_length': 20,
#     'save_dir': None,  # 可以设置保存目录
#     'save_interval': 100,
#     'enable_export': False  # 训练时可以关闭导出
#   }
#   # initiate the tracker
#   # tracker = DMSTrack(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start,
#   #                    dtype=dtype, device=device,
#   #                    differentiable_kalman_filter_config=differentiable_kalman_filter_config,
#   #                    observation_covariance_net_dict=observation_covariance_net_dict,
#   #                    force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R)
#
#   # tracker = BestForNowTrackV1(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,
#   #                     enable_learnable_Q=enable_learnable_Q)
#   # tracker = EnhancedDMSTrack(cfg, cat, q_learning_config, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None,train=True,innovation_based_q_net=innovation_based_q_net)
#   tracker = BestForNowTrack(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,enable_learnable_Q=enable_learnable_Q,  q_learning_config=q_learning_config ,q_net_dict=q_net_dict)
#   # tracker = ParallelEnhancedDMSTrack(cfg, cat, q_learning_config, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None,train=False,innovation_based_q_net=innovation_based_q_net,enable_learnable_Q=enable_learnable_Q)
#   # tracker = SeriallDMSTrackWithCI(cfg, cat, q_learning_config, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None,train=False,innovation_based_q_net=innovation_based_q_net,enable_learnable_Q=enable_learnable_Q)
#   # ====================设置过程噪声网络=======================
#   # if enable_learnable_Q and q_net:
#   #   tracker.process_noise_net = q_net
#     # if use_multiple_nets:
#     #   # 这里需要决定使用哪个CAV的网络，可以使用ego的
#     #   if 'ego' in innovation_based_q_net:
#     #     tracker.process_noise_net = innovation_based_q_net['ego']
#     # else:
#     #   if 'shared' in innovation_based_q_net:
#     #     tracker.process_noise_net = innovation_based_q_net['shared']
#   # tracker = DMSTrackWithHistory(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None)
#   # MY_CODE:
#   # seq_name is one of ['0000', '0001', ... '0008']
#   # use it to read tracking label data to get the frames
#   # ~/my_cooperative_tracking/AB3DMOT/scripts/KITTI/v2v4real_label/0000.txt
#   tracking_label_file = '../AB3DMOT/scripts/KITTI/v2v4real_%s_label/%s.txt' % (split, seq_name)
#   with open(tracking_label_file, 'r') as f:
#     for line in f:
#       frame_id = int(line.split(' ')[0])
#   frame_list = list(range(frame_id + 1))
#   #print('frame_list: ', frame_list)
#
#   return tracker, frame_list
#
#
# def get_len_per_seq_eval(len_record, seq_eval):
#   '''
#   Input
#     len_record: all accumulated length of seqs
#     seq_eval: list of seq_name being considered: ['0003'] or ['0000', '0001', ...]
#   Output
#     len_per_seq_eval: length of seq in seq_eval: val split example [198] or [147, 114, 144, ...]
#   '''
#   len_per_seq = [len_record[0]]
#   for i in range(1, len(len_record)):
#     len_per_seq.append(len_record[i] - len_record[i-1])
#   #print('len_record: ', len_record)
#   #print('len_per_seq: ', len_per_seq)
#   if len(seq_eval) == 1:
#     len_per_seq_eval = [len_per_seq[int(seq_eval[0])]]
#   else:
#     len_per_seq_eval = len_per_seq
#   #print('len_per_seq_eval: ', len_per_seq_eval)
#   return len_per_seq_eval
#
#
# def get_sub_seq_list_info(is_training, num_frames_per_sub_seq, len_record, len_per_seq_eval, seq_eval):
#   '''
#   Divide all sequences of a epoch into list of sub sequences.
#   If is_training is True, randome shuffle and each sub seq has length num_frames_per_sub_seq.
#   if num_frames_per_sub_seq == -1 or is_training is False,
#   No random shuffle, each sequence is a sub sequence, list length is number of sequences
#
#   Input:
#     is_training
#     num_frames_per_sub_seq : length of each sub seq if is_training is True
#     len_record: all accumulated length of seqs: [147, 114, 144, 198, 180, 310, 304, 221, 375]
#     len_per_seq_eval: length per seq in seq_eval: [198] or [147, 114, 144, 198, ...]
#     seq_eval: list of seq_name being considered: ['0003'] or ['0000', '0001', ...]
#
#   Output
#     sub_seq_list_info: numpy [num_sub_seqs, 3]
#       3:
#         seq_name_list: global seq name
#         seq_idx_in_eval_list: seq idx in seq_eval
#         start_frame_idx_list: start frame idx in a seq
#         end_frame_idx_list: end frame idx in a seq
#   '''
#   #print('num_frames_per_sub_seq: ', num_frames_per_sub_seq)
#   #print('seq_eval: ', seq_eval)
#   #print('len_record: ', len_record)
#   #print('len_per_seq_eval: ', len_per_seq_eval)
#
#   # not used
#   #seq_id_from_seq_eval_name = [int(id) for id in seq_eval]
#   #print('seq_id_from_seq_eval_name: ', seq_id_from_seq_eval_name)
#
#   num_seqs_eval = len(seq_eval)
#
#   if not is_training or num_frames_per_sub_seq == -1:
#     num_sub_seqs = num_seqs_eval
#     seq_name_list = seq_eval
#     seq_idx_in_eval_list = list(range(num_seqs_eval))
#     start_frame_idx_list = [0] * num_seqs_eval
#     end_frame_idx_list = [length-1 for length in len_per_seq_eval]
#   else:
#     seq_name_list = []
#     seq_idx_in_eval_list = []
#     start_frame_idx_list = []
#
#     overlapping_sliding_window = False
#     if overlapping_sliding_window:
#       for i in range(len(seq_eval)):
#         seq_name_list += [seq_eval[i]] * (len_per_seq_eval[i] - (num_frames_per_sub_seq- 1))
#         seq_idx_in_eval_list += [i] * (len_per_seq_eval[i] - (num_frames_per_sub_seq - 1))
#         start_frame_idx_list += list(range(len_per_seq_eval[i] - (num_frames_per_sub_seq - 1)))
#       end_frame_idx_list = [start + num_frames_per_sub_seq - 1 for start in start_frame_idx_list]
#     else:
#       for i in range(len(seq_eval)):
#         print('len_per_seq_eval[i]: ', len_per_seq_eval[i])
#         print('num_frames_per_sub_seq: ', num_frames_per_sub_seq)
#         print('len_per_seq_eval[i] // num_frames_per_sub_seq: ', len_per_seq_eval[i] // num_frames_per_sub_seq)
#         num_sub_seqs_in_this_seq = len_per_seq_eval[i] // num_frames_per_sub_seq
#         seq_name_list += [seq_eval[i]] * num_sub_seqs_in_this_seq
#         seq_idx_in_eval_list += [i] * num_sub_seqs_in_this_seq
#         start_frame_idx_list += [j * num_frames_per_sub_seq for j in range(num_sub_seqs_in_this_seq)]
#       end_frame_idx_list = [start + num_frames_per_sub_seq - 1 for start in start_frame_idx_list]
#
#     assert(len(seq_name_list) == len(start_frame_idx_list))
#     assert(len(seq_idx_in_eval_list) == len(start_frame_idx_list))
#
#   print('seq_name_list: ', seq_name_list)
#   print('seq_idx_in_eval_list: ', seq_idx_in_eval_list)
#   print('start_frame_idx_list: ', start_frame_idx_list)
#   print('end_frame_idx_list: ', end_frame_idx_list)
#
#
#   return seq_name_list, seq_idx_in_eval_list, start_frame_idx_list, end_frame_idx_list
#
# def save_frame_tracking_results(save_dir, seq_name, frame_data_list):
#   '''
#   保存单个序列的所有帧数据
#
#   frame_data_list: list of dict, 每个dict包含一帧的数据:
#       {
#           'frame': int,
#           'tracking_results': np.array,
#           'gt_boxes': np.array,
#           'gt_ids': np.array,
#           'track_P': list
#       }
#   '''
#   import json
#
#   # JSON格式（易读）
#   json_file = os.path.join(save_dir, f'seq_{seq_name}_summary.json')
#   summary = {
#     'sequence': seq_name,
#     'total_frames': len(frame_data_list),
#     'frames': []
#   }
#
#   for data in frame_data_list:
#     frame_info = {
#       'frame_id': int(data['frame']),
#       'num_tracks': len(data['tracking_results']),
#       'num_gt': len(data['gt_boxes']),
#       'tracks': [],
#       'ground_truth': []
#     }
#
#     # 跟踪结果
#     for trk in data['tracking_results']:
#       frame_info['tracks'].append({
#         'id': int(trk[7]),
#         'position': {'x': float(trk[3]), 'y': float(trk[4]), 'z': float(trk[5])},
#         'size': {'h': float(trk[0]), 'w': float(trk[1]), 'l': float(trk[2])},
#         'rotation': float(trk[6]),
#         'score': float(trk[14])
#       })
#
#     # Ground truth
#     for gt_box, gt_id in zip(data['gt_boxes'], data['gt_ids']):
#       frame_info['ground_truth'].append({
#         'id': int(gt_id),
#         'position': {'x': float(gt_box[0]), 'y': float(gt_box[1]), 'z': float(gt_box[2])},
#         'size': {'h': float(gt_box[6]), 'w': float(gt_box[5]), 'l': float(gt_box[4])},
#         'rotation': float(gt_box[3])
#       })
#
#     summary['frames'].append(frame_info)
#
#   with open(json_file, 'w') as f:
#     json.dump(summary, f, indent=2)
#
#   print(f"[Info] Saved summary to {json_file}")
#
# def train_one_epoch(
#     is_training, loss_types, writer, step_index,
#     save_tracking_results,
#     seq_eval, cav_id_list,
#     det_root, eval_dir_dict, save_dir, result_sha,
#     cfg, cat, ID_start, hw, log,
#     dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict,
#     optimizer, len_record, detection_feature_path, training_ground_truth_data_all, det_id2str,
#     num_frames_backprop, num_frames_per_sub_seq, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis, training_split,
#     regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, clip_grad_norm, parameters_list,q_learning_config
#     , enable_learnable_Q=False, q_net_dict=None,
#     trajectory_q_loss_weight=None,  # 新增参数
#     rq_consistency_loss_weight=None
# ):
#   '''
#   Train one epoch
#   When is_training is False, then optimizer is None,
#   this function is called by evaluation, num_frames_backprop = num_frames_per_sub_seq = -1
#   and we do not call loss.backprop() or optimizer.step(),
#   only finish tracking and save the results
#
#   Output:
#     loss_stats_avg: average loss over all seqs of one epoch
#     loss_stats_avg_dict: details of different types of loss
#   '''
#   if not is_training:
#     assert(num_frames_backprop == -1)
#     assert(num_frames_per_sub_seq == -1)
#
#   len_per_seq_eval = get_len_per_seq_eval(len_record, seq_eval)
#   # During training, each sub seq has length == num_frames_backprop
#   # During inference, each sub seq is a seq
#   seq_name_list, seq_idx_in_eval_list, start_frame_idx_list, end_frame_idx_list = get_sub_seq_list_info(is_training, num_frames_per_sub_seq, len_record, len_per_seq_eval, seq_eval)
#   num_sub_seqs = len(seq_name_list)
#   print('num_sub_seqs: ', num_sub_seqs)
#
#
#   # loss for stats over one epoch's full sequence
#   loss_stats_list_dict = {
#     loss_type: {
#       'sum': [],
#       'count': 0
#     } for loss_type in loss_types
#   }
#
#   # ========== 添加这里 ==========
#   # 用于收集每个序列的跟踪数据
#   sequence_frame_data = []
#   current_seq_name = None
#   # ================================
#
#   # loop every sub sequence
#   sub_seq_count = 0
#   total_time, total_frames = 0.0, 0
#
#   # random shuffle if training
#   if is_training:
#     sample_sub_seq_indices = np.random.permutation(num_sub_seqs)
#   else:
#     # no random shuffle
#     sample_sub_seq_indices = range(num_sub_seqs)
#
#   #for seq_idx_in_eval in range(len(seq_eval)):
#   #for sub_seq_idx in range(num_sub_seqs):
#   for sub_seq_idx in sample_sub_seq_indices:
#     #print('sub_seq_idx: ', sub_seq_idx)
#
#     #seq_name = seq_eval[seq_idx_in_eval]
#     seq_name = seq_name_list[sub_seq_idx]
#     seq_idx_in_eval = seq_idx_in_eval_list[sub_seq_idx]
#     # ========== 添加这里：检测序列切换 ==========
#     if current_seq_name is not None and current_seq_name != seq_name:
#       # 保存上一个序列的数据
#       # if save_vis and len(sequence_frame_data) > 0:
#       if save_vis and len(sequence_frame_data) > 0:
#         save_vis_dir = os.path.join(save_dir, 'visualization')
#         save_frame_tracking_results(save_vis_dir, current_seq_name, sequence_frame_data)
#         sequence_frame_data = []  # 清空
#
#     current_seq_name = seq_name
#     # ============================================
#     # if seq_name == '0000' or seq_name == '0001' or seq_name == '0002' or seq_name == '0003' or seq_name == '0004' or seq_name == '0005' or seq_name == '0006':  # 跳过第一个场景
#     #     continue
#     seq_dets_dict = {}
#     for cav_id in cav_id_list:
#       seq_file = os.path.join(det_root, cav_id, seq_name+'.txt')
#       #print('seq_file: ', seq_file)
#       seq_dets, flag = load_detection(seq_file)         # load detection
#       #print('flag: ', flag)
#       if flag:
#         seq_dets_dict[cav_id] = seq_dets
#     if len(seq_dets_dict) == 0:
#       continue                  # no detection
#     #print('seq_dets_dict: ', seq_dets_dict)
#
#     # create folders for saving
#     eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = \
#       get_saving_dir(eval_dir_dict, seq_name, save_dir, cfg.num_hypo)
#
#     # initialize tracker for training
#     tracker, frame_list = initialize_DMSTrack(cfg, seq_name, cat, ID_start, hw, log,
#       dtype, device,
#       differentiable_kalman_filter_config,
#       observation_covariance_net_dict,
#       force_gt_as_predicted_track,
#       use_static_default_R,
#       use_multiple_nets,
#       training_split,
#       q_learning_config,
#       enable_learnable_Q=enable_learnable_Q,
#       q_net_dict=q_net_dict)
#     #print('frame_list: ', frame_list)
#
#     # reset optimizer and loss
#     if is_training:
#       optimizer.zero_grad()
#     # actual loss for backprop
#     loss_list_dict = {
#       loss_type: {
#         'sum': [],
#         'count': 0
#       } for loss_type in loss_types
#     }
#
#
#     # loop over frame
#     #min_frame, max_frame = int(frame_list[0]), int(frame_list[-1])
#     min_frame = start_frame_idx_list[sub_seq_idx]
#     max_frame = end_frame_idx_list[sub_seq_idx]
#     #print('min_frame: ', min_frame)
#     #print('max_frame: ', max_frame)
#
#     # MY_DEBUG
#     # max_frame = min_frame
#     start_time = time.time()
#     # tracer.start()
#     for frame in range(min_frame, max_frame + 1):
#     # for frame in range(min_frame, min_frame + 4):
#     # for frame in range(0, 2):
#       # for debug
#       #print('frame: ', frame)
#       #if frame == 2:
#       #  break
#
#       # add an additional frame here to deal with the case that the last frame, although no detection
#       # but should output an N x 0 affinity for consistency
#       # start = time.time()
#       # print progress during inference
#       if not is_training:
#         print_str = 'processing %s %s: %d/%d, %d/%d   \r' % (result_sha, seq_name, sub_seq_count, \
#           num_sub_seqs, frame, max_frame)
#         sys.stdout.write(print_str)
#         sys.stdout.flush()
#
#       # for accessing single frame object feature of v2v4real
#       global_timestamp_index = get_global_timestamp_index(seq_name, frame, len_record)
#
#       # tracking by detection
#       dets_frame_dict = {}
#       dets_feature_dict = {}
#       transformation_matrix_dict = {}
#       for cav_id in seq_dets_dict.keys():
#         dets_frame_dict[cav_id] = get_frame_det(seq_dets_dict[cav_id], frame) # 处理指定列数据（1-6列；7-13列：包括物体的3D中心坐标 (x, y, z)、尺寸 (宽, 高, 长) 和旋转角度）
#         if 'multi_sensor_differentiable_kalman_filter' in cfg.det_name:
#           dets_feature_dict[cav_id] = load_detection_feature(global_timestamp_index, detection_feature_path, cav_id)
#         else:
#           dets_feature_dict[cav_id] = load_detection_feature_cobevt_single_sensor(global_timestamp_index, detection_feature_path, cav_id)
#         transformation_matrix_dict[cav_id] = load_transformation_matrix(global_timestamp_index, detection_feature_path, cav_id)
#
#       # get ground-truth tensor data for this frame during training
#       gt_boxes, gt_ids = get_ground_truth_data_single_frame(seq_idx_in_eval, frame, training_ground_truth_data_all, dtype, device)
#
#       since = time.time()
#       #print('1 cav_id_list: ', cav_id_list)
#       results, affi, loss_dict, matched_detection_id_dict, learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict = tracker.track_multi_sensor_differentiable_kalman_filter(
#         dets_frame_dict, frame, seq_name, cav_id_list, dets_feature_dict, gt_boxes, gt_ids, transformation_matrix_dict)
#       # results, affi, loss_dict, matched_detection_id_dict, learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict = tracker.track_multi_agent_with_mix_ci(
#       #   dets_frame_dict, frame, seq_name, cav_id_list, dets_feature_dict, gt_boxes, gt_ids, transformation_matrix_dict)
#       total_time += time.time() - since
#       # print(total_time)
#       # print('loss_dict: ', loss_dict)
#       for loss_type in loss_types:
#         # actual loss
#         loss_list_dict[loss_type]['sum'].append(loss_dict[loss_type]['sum'])
#         loss_list_dict[loss_type]['count'] += loss_dict[loss_type]['count']
#         # loss stats
#         loss_stats_list_dict[loss_type]['sum'].append(loss_dict[loss_type]['sum'].detach().cpu().numpy())
#         loss_stats_list_dict[loss_type]['count'] += loss_dict[loss_type]['count']
#
#       #print('loss_stats_list_dict: ', loss_stats_list_dict)
#
#       # loss backprop and optimize
#       loss_backprop_at_this_frame = False
#       # comment out old code
#       #if num_frames_backprop == -1 or frame == max_frame:
#       #  if frame == max_frame:
#       #    loss_backprop_at_this_frame = True
#       #else:
#       #  if frame != 0 and (frame % num_frames_backprop == 0 or frame == max_frame):
#       #    loss_backprop_at_this_frame = True
#
#       # loss backprop at end of each sub seq
#       if frame == max_frame:
#         loss_backprop_at_this_frame = True
#       # or we reach the targeted number of frames
#       if  (frame - min_frame + 1) % num_frames_backprop == 0:
#         loss_backprop_at_this_frame = True
#
#       # if called by evaluate(), do not backprop
#       if not is_training:
#         loss_backprop_at_this_frame = False
#
#       # MY_DEBUG
#       # force no learning
#       #loss_backprop_at_this_frame = False
#       #print('loss_backprop_at_this_frame: ', loss_backprop_at_this_frame)
#       #assert False
#
#       #print('loss_backprop_at_this_frame: ', loss_backprop_at_this_frame)
#
#       if loss_backprop_at_this_frame:
#         loss_avg_dict = {}
#         for loss_type in loss_types:
#           # loss_all = torch.stack(loss_list_dict[loss_type]['sum'])
#           # #print('loss_all: ', loss_all)
#           # loss_sum = torch.sum(loss_all)
#           loss_sum = torch.tensor(0.0, dtype=dtype, device=device)
#           for tensor in loss_list_dict[loss_type]['sum']:
#             loss_sum += torch.sum(tensor)  # torch.sum()能处理任意形状的张量
#           #print('loss_sum: ', loss_sum)
#           if loss_list_dict[loss_type]['count'] == 0:
#             loss_list_dict[loss_type]['count'] += 1
#           loss_avg = loss_sum / loss_list_dict[loss_type]['count']
#           #print('loss_avg: ', loss_avg)
#           loss_avg_dict[loss_type] = loss_avg
#
#         regression_loss_avg = loss_avg_dict['regression']
#         association_loss_avg = loss_avg_dict['association']
#         det_neg_log_likelihood_loss_avg = loss_avg_dict['det_neg_log_likelihood']
#         # ====================添加trajectory_q_loss损失和trajectory_s_loss损失==================================
#         if enable_learnable_Q and ('trajectory_q_loss' in loss_avg_dict or 'trajectory_s_loss' in loss_avg_dict):
#             trajectory_q_loss_avg = loss_avg_dict.get('trajectory_q_loss',
#                                                       torch.tensor(0.0, dtype=dtype, device=device))
#             trajectory_s_loss = loss_avg_dict.get('trajectory_s_loss', torch.tensor(0.0, dtype=dtype, device=device))
#
#             # 总Q损失权重（可以调整）
#
#
#             loss_avg = (regression_loss_avg * regression_loss_weight +
#                         association_loss_avg * association_loss_weight +
#                         det_neg_log_likelihood_loss_avg * det_neg_log_likelihood_loss_weight +
#                         trajectory_q_loss_avg * trajectory_q_loss_weight +
#                         trajectory_s_loss * rq_consistency_loss_weight)
#
#             print(
#                 'frame: %d, actual_loss: %f, det_neg_log_likelihood_loss: %f, regression_loss: %f, association_loss: %f, trajectory_q_loss: %f, trajectory_s_loss: %f' %
#                 (frame, loss_avg.item(), det_neg_log_likelihood_loss_avg.item(),
#                  regression_loss_avg.item(), association_loss_avg.item(),
#                  trajectory_q_loss_avg.item(), trajectory_s_loss.item()))
#
#
#         else:
#           loss_avg = (regression_loss_avg * regression_loss_weight +
#                       association_loss_avg * association_loss_weight +
#                       det_neg_log_likelihood_loss_avg * det_neg_log_likelihood_loss_weight)
#
#           print(
#             'frame: %d, actual_loss: %f, det_neg_log_likelihood_loss: %f, regression_loss: %f, association_loss: %f' %
#             (frame, loss_avg.item(), det_neg_log_likelihood_loss_avg.item(),
#              regression_loss_avg.item(), association_loss_avg.item()))
#
#
#         tensorboard_loss_dict = {loss_type: loss_avg_dict[loss_type].item() for loss_type in loss_types}
#         write_tensorboard(writer, 'step', step_index, loss_types,
#           None, loss_avg.item(), tensorboard_loss_dict,
#           None, None, None)
#
#         if not use_static_default_R:
#           loss_avg.backward()
#           if clip_grad_norm > 0:
#             torch.nn.utils.clip_grad_norm_(parameters_list, clip_grad_norm)
#           optimizer.step()
#           step_index += 1
#         # reset
#         # reset optimizer and loss
#         optimizer.zero_grad()
#         loss_list_dict = {
#           loss_type: {
#             'sum': [],
#             'count': 0
#           } for loss_type in loss_types
#         }
#
#         # reset dkf model gradients
#         tracker.reset_dkf_gradients()
#
#
#
#       if save_tracking_results:
#         # saving affinity matrix, between the past frame and current frame
#         # e.g., for 000006.npy, it means affinity between frame 5 and 6
#         # note that the saved value in affinity can be different in reality because it is between the
#         # original detections and ego-motion compensated predicted tracklets, rather than between the
#         # actual two sets of output tracklets
#         save_affi_file = os.path.join(affinity_dir, '%06d.npy' % frame)
#         save_affi_vis  = os.path.join(affinity_vis, '%06d.txt' % frame)
#         if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0):
#           # save affinity as long as there are tracklets in at least one frame
#           np.save(save_affi_file, affi)
#           # cannot save for visualization unless both two frames have tracklets
#           if affi.shape[0] > 0 and affi.shape[1] > 0:
#             save_affinity(affi, save_affi_vis)
#         # saving trajectories, loop over each hypothesis
#         for hypo in range(cfg.num_hypo):
#           save_trk_file = os.path.join(save_trk_dir[hypo], '%06d.txt' % frame)
#           save_trk_file = open(save_trk_file, 'w')
#           for result_tmp in results[hypo]:        # N x 15
#             save_results(result_tmp, save_trk_file, eval_file_dict[hypo], \
#               det_id2str, frame, cfg.score_threshold)
#           save_trk_file.close()
#
#       if show_vis or save_vis:
#         if save_vis:
#           sequence_frame_data.append({
#             'frame': frame,
#             'tracking_results': results[0].copy(),  # 假设使用第一个hypothesis
#             'gt_boxes': gt_boxes.detach().cpu().numpy(),
#             'gt_ids': gt_ids.detach().cpu().numpy(),
#             'track_P': track_P
#           })
#         if seq_name == '0008':
#           if save_vis and len(sequence_frame_data) > 0:
#             save_vis_dir = os.path.join(save_dir, 'visualization')
#             save_frame_tracking_results(save_vis_dir, current_seq_name, sequence_frame_data)
#           assert(len(results) == 1)
#         # MY_DEBUG
#         if True:
#           # 删除特定帧的特定ID
#           ids_to_delete = {
#             101: [26],
#             102: [26,43]# 第101帧删除ID为3和5的跟踪框
#
#           }
#           manual_adjustments = {
#             # 32: {  # 第101帧
#             #   # 1: {'dx': 0.0, 'dy': -1.0, 'dz': 0.0, 'dtheta': 0.1},
#             #   0: {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1},
#             #   # 10: {'dx': 2.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1}
#             # },
#             # 33: {  # 第102帧
#             #   0: {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1},
#             #   # 10: {'dx': 2.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1}
#             # }
#           }
#         #if frame >=15:
#           save_vis_dir = os.path.join(save_dir, 'visualization')
#           mkdir_if_missing(save_vis_dir)
#           visualize(
#             show_vis, save_vis, save_vis_dir,
#             results[0],
#             gt_boxes.detach().cpu().numpy(),
#             gt_ids.detach().cpu().numpy(),
#             dets_frame_dict,
#             detection_feature_path, seq_name, frame, global_timestamp_index,
#             matched_detection_id_dict, learnable_R_dict, track_P,manual_adjustments=manual_adjustments,ids_to_delete=ids_to_delete)
#
#
#
#       # end of this frame
#       sys.stdout.flush()
#       total_frames += 1
#     # tracer.stop()
#     # tracer.save()
#     end =time.time()
#     print('frame: %d, total time: %f' % (total_frames, end-start_time))
#     # tracer.stop()
#     # tracer.save()
#     #
#     # end of this sub seq
#     # tracer.stop()
#     # tracer.save()
#     sys.stdout.flush()
#     sub_seq_count += 1
#
#     for index in range(cfg.num_hypo):
#       eval_file_dict[index].close()
#       ID_start = max(ID_start, tracker.ID_count[index])
#
#     #print_log('%s, %25s: %4.f seconds for %5d frames or %6.1f FPS, metric is %s = %.2f' % \
#     #  (cfg.dataset, result_sha, total_time, total_frames, total_frames / total_time, \
#     #  tracker.metric, tracker.thres), log=log)
#
#
#   speed = total_frames / total_time
#   print('Tracking speed: %f frames per second, total frames: %d, total time: %f' % (speed, total_frames, total_time))
#   #assert False
#
#
#   # print loss stats at the end of epoch
#   # MY_DEBUG
#   #if frame == max_frame:
#   if True:
#     loss_stats_avg_dict = {}
#     for loss_type in loss_types:
#       # 确保所有损失值都是标量
#       loss_stats_sum_list = []
#       for loss_val in loss_stats_list_dict[loss_type]['sum']:
#         # 强制转换为标量
#         if hasattr(loss_val, 'item'):
#           scalar_val = loss_val.item()
#         elif np.isscalar(loss_val):
#           scalar_val = float(loss_val)
#         else:
#           scalar_val = float(np.sum(loss_val))  # 如果是数组，求和后转标量
#         loss_stats_sum_list.append(scalar_val)
#       loss_stats_sum = np.sum(loss_stats_sum_list)
#       # loss_stats_all = np.stack(loss_stats_list_dict[loss_type]['sum'])
#       # loss_stats_sum = np.sum(loss_stats_all)
#       if loss_stats_list_dict[loss_type]['count'] == 0:
#         loss_stats_list_dict[loss_type]['count'] += 1
#       loss_stats_avg = loss_stats_sum / loss_stats_list_dict[loss_type]['count']
#       loss_stats_avg_dict[loss_type] = loss_stats_avg
#
#     regression_loss_stats_avg = loss_stats_avg_dict['regression']
#     association_loss_stats_avg = loss_stats_avg_dict['association']
#     det_neg_log_likelihood_loss_stats_avg = loss_stats_avg_dict['det_neg_log_likelihood']
#     if enable_learnable_Q:
#       trajectory_q_loss_stats_avg = loss_stats_avg_dict['trajectory_q_loss']
#       rq_consistency_loss_stats_avg = loss_stats_avg_dict['trajectory_s_loss']
#     # process_noise_loss_avg = loss_stats_avg_dict['process_noise']
#     if enable_learnable_Q:
#       loss_stats_avg = regression_loss_stats_avg * regression_loss_weight + association_loss_stats_avg * association_loss_weight + det_neg_log_likelihood_loss_stats_avg * det_neg_log_likelihood_loss_weight + trajectory_q_loss_stats_avg * trajectory_q_loss_weight + rq_consistency_loss_stats_avg * rq_consistency_loss_weight
#       print('frame: %d,  loss_stats: %f, det_neg_log_likelihood_loss: %f, regression_loss: %f, association_loss: %f，trajectory_q_loss: %f, trajectory_s_loss: %f' % (frame, loss_stats_avg.item(), det_neg_log_likelihood_loss_stats_avg.item(), regression_loss_stats_avg.item(), association_loss_stats_avg.item(), trajectory_q_loss_stats_avg.item(), rq_consistency_loss_stats_avg.item()))
#
#     else:
#       loss_stats_avg = regression_loss_stats_avg * regression_loss_weight + association_loss_stats_avg * association_loss_weight + det_neg_log_likelihood_loss_stats_avg * det_neg_log_likelihood_loss_weight
#       print('frame: %d,  loss_stats: %f, det_neg_log_likelihood_loss: %f, regression_loss: %f, association_loss: %f' % (frame, loss_stats_avg.item(), det_neg_log_likelihood_loss_stats_avg.item(), regression_loss_stats_avg.item(), association_loss_stats_avg.item()))
#
#     # reset
#     #loss_stats_list_dict = {
#     #  loss_type: {
#     #    'sum': [],
#     #    'count': 0
#     #  } for loss_type in loss_types
#     #}
#   # end of this epoch
#   sys.stdout.flush()
#
#   return loss_stats_avg, loss_stats_avg_dict, step_index
#
#
# def get_evaluation_metrics(evaluation_file_save_dir):
#   print('evaluation_file_save_dir: ', evaluation_file_save_dir)
#   # ./results/v2v4real/debug_evaluation_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_0
#   evaluation_summary_file = os.path.join(evaluation_file_save_dir, 'summary_car_average_eval3D.txt')
#   metrics_dict = {}
#
#   # get the line right below ' sAMOTA  AMOTA  AMOTP'
#   found_metrics_line = False
#   with open(evaluation_summary_file, 'r') as f:
#     for line in f:
#       line = line.strip()
#       print(line)
#       if found_metrics_line:
#         metrics_dict['sAMOTA'] = float(line[0])
#         metrics_dict['AMOTA'] = float(line[1])
#         metrics_dict['AMOTP'] = float(line[2])
#         break
#
#       if len(line) > 0 and line[0] == 'sAMOTA':
#         found_metrics_line = True
#
#   print('metrics_dict: ', metrics_dict)
#   assert False
#   return metrics_dict
#
#
# # Using the model to run tracking and evaluation on the evaluation split, without loss backprop
# def track_and_evaluate(save_dir_prefix, cav_id_list, observation_covariance_net_dict, evaluation_ground_truth_data_all, evaluation_config_dict, evaluation_split, cfg, seq_eval_mode, epoch_idx, cat, ID_start, hw, log, dtype, device, differentiable_kalman_filter_config, det_id2str, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis, mail, threshold_3D_IOU, loss_types, writer, regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight,
#                        q_learning_config,enable_learnable_Q=False, q_net_dict=None, trajectory_q_loss_weight=None, rq_consistency_loss_weight=None):
#   '''
#   Similar to the code calling train_one_epoch, but do not call loss.backprop() and optimizer.step()
#   '''
#   with torch.no_grad():
#     for cav_id, model in observation_covariance_net_dict.items():
#       model.eval()
#
#     print('epoch_idx: ', epoch_idx)
#
#     evaluation_save_folder = 'evaluation_' + evaluation_config_dict['result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % epoch_idx
#     evaluation_save_folder = os.path.join(save_dir_prefix, evaluation_save_folder)
#     evaluation_save_dir = os.path.join(cfg.save_root, evaluation_save_folder); mkdir_if_missing(evaluation_save_dir)
#     print('evaluation_save_dir: ', evaluation_save_dir)
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_all_H1_epoch_0
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_0001_H1_epoch_0
#
#     # create eval dir for each hypothesis
#     eval_dir_dict = dict()
#     for index in range(cfg.num_hypo):
#       eval_dir_dict[index] = os.path.join(evaluation_save_dir, 'data_%d' % index); mkdir_if_missing(eval_dir_dict[index])
#     #print('eval_dir_dict[0]: ', eval_dir_dict[0])
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0/data_0
#
#     save_tracking_results = True
#     # set is_training = False, when calling train_one_epoch
#     # optimizer = None
#     # also set num_frames_backprop = -1
#     # num_frames_per_sub_seq = -1
#     # set step_index = -1 because optimizer.step() will not be called
#     # the returned step_index will not be used
#     # set clip_grad_norm = -1, we will not call optimizer.step()
#     # set parameters_list = []
#     evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict, _ = train_one_epoch(
#       False, loss_types, writer, -1,
#       save_tracking_results,
#       evaluation_config_dict['seq_eval'], cav_id_list,
#       evaluation_config_dict['det_root'], eval_dir_dict, evaluation_save_dir, evaluation_config_dict['result_sha'],
#       cfg, cat, ID_start, hw, log,
#       dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict,
#       None, evaluation_config_dict['len_record'], evaluation_config_dict['detection_feature_path'],
#       evaluation_ground_truth_data_all, det_id2str,
#       -1, -1, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis, evaluation_split,
#       regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, -1, [],q_learning_config,
#       enable_learnable_Q, q_net_dict,
#       trajectory_q_loss_weight,  # 新增参数
#       rq_consistency_loss_weight, )
#
#     # =================保存过程噪声网络检查点===============================
#     if enable_learnable_Q:
#
#       for cav_id, model in q_net_dict.items():
#         # model.eval()
#         trajectory_q_net_model_weight = os.path.join(evaluation_save_dir, 'trajectory_q_net_model%s_epoch_%d.pth' % (cav_id, epoch_idx))
#         torch.save(q_net_dict[cav_id].state_dict(), trajectory_q_net_model_weight)
#
#     # save model checkpoint
#     for cav_id in cav_id_list:
#       model_save_file = os.path.join(evaluation_save_dir, 'model_%s_epoch_%d.pth' % (cav_id, epoch_idx))
#       torch.save(observation_covariance_net_dict[cav_id].state_dict(), model_save_file)
#
#
#
#     run_evaluation_every_epoch = True
#     if run_evaluation_every_epoch:
#       # 'all', '0000', '0001', ...
#       # this command will evaluate the current evaluation split's tracking results
#       #evaluation_split_evaluation_command = 'python3 scripts/KITTI/evaluate.py ' + evaluation_save_folder + ' 1 3D 0.25 %s %s' % (seq_eval_mode, evaluation_split)
#       #print('evaluation_split_evaluation_command: ', evaluation_split_evaluation_command)
#       #returned_value = os.system(evaluation_split_evaluation_command)
#       #print('returned_value: ', returned_value)
#
#       # directly call ab3dmot evaluate()
#       evaluation_metrics_dict = evaluate(evaluation_save_folder, mail, 1, True, False, threshold_3D_IOU, True, seq_eval_mode, evaluation_split)
#
#
#     for cav_id, model in observation_covariance_net_dict.items():
#       model.train()
#
#   sys.stdout.flush()
#   return evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict
#
#
#
# def get_config(cfg, cat, training_split, evaluation_split, seq_eval_mode, use_reversed_cav_id_list):
#   '''
#   Input:
#     cfg.det_name
#     cfg.data_set
#     training_split: can be 'train' or 'val'
#     evaluation_split: can be 'train' or 'val'
#     seq_eval_mode: one of ['all', '0001', '0002', ...]
#
#   Output:
#
#     training_config_dict = {
#       result_sha :
#       det_root :
#       seq_eval :
#       len_record :
#       detection_feature_path :
#     }
#
#     evaluation_config_dict has the same structure as above
#
#     det_id2str
#     hw
#     data_root
#     cav_id_list
#     differentiable_kalman_filter_config
#     learning_rate
#
#   '''
#   # common config
#   det_id2str = {2: 'Car'}
#   hw = {'image': None, 'lidar': None}
#   data_root = '../AB3DMOT/data/v2v4real'
#   if use_reversed_cav_id_list:
#     cav_id_list =  ['1', 'ego']
#   else:
#     cav_id_list =  ['ego', '1']
#
#   if 'multi_sensor_differentiable_kalman_filter' not in cfg.det_name:
#     # single sensor, such as cobevt + dkf
#     cav_id_list = ['ego']
#
#   if 'multi_sensor_differentiable_kalman_filter' in cfg.det_name:
#     dkf_type = 'multi_sensor'
#     feature_channel_size = 320
#     feature_region_size = 20
#   else: # cobevt + dkf
#     dkf_type = 'single_sensor'
#     feature_channel_size = 512
#     feature_region_size = 5
#   differentiable_kalman_filter_config = {
#     'dim_x': 10,
#     'dim_z': 7,
#     'dkf_type': dkf_type,
#     'observation_covariance_setting': {
#       'feature_channel_size': feature_channel_size,
#       'feature_region_size': feature_region_size
#     },
#     'gt_data_association_threshold': 4 # 4 meters center distance
#   }
#   learning_rate = 1e-3
#   weight_decay = 1e-5
#
#   train_result_sha = '%s_%s_%s' % (cfg.det_name, cat, 'train')
#   train_det_root = os.path.join('../AB3DMOT/data', cfg.dataset, 'detection', train_result_sha)
#
#   val_result_sha = '%s_%s_%s' % (cfg.det_name, cat, 'val')
#   val_det_root = os.path.join('../AB3DMOT/data', cfg.dataset, 'detection', val_result_sha)
#
#   train_seq_eval = [
#     '0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',
#     '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
#     '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029',
#     '0030', '0031'
#   ]
#   train_len_record = [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
#
#   val_seq_eval = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
#   val_len_record = [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
#
#   if 'multi_sensor_differentiable_kalman_filter' in cfg.det_name:
#     train_detection_feature_path = '../V2V4Real/official_models/train_no_fusion_keep_all/npy/'
#     val_detection_feature_path = '../V2V4Real/official_models/no_fusion_keep_all/npy/'
#   else: # cobevt + dkf
#     train_detection_feature_path = '../V2V4Real/official_models/train_cobevt/npy/'
#     val_detection_feature_path = '../V2V4Real/official_models/cobevt/npy/'
#
#
#   train_config_dict = {
#     'result_sha' : train_result_sha,
#     'det_root' : train_det_root,
#     'seq_eval' : train_seq_eval if seq_eval_mode == 'all' else [seq_eval_mode],
#     'len_record' : train_len_record,
#     'detection_feature_path' :  train_detection_feature_path
#   }
#   val_config_dict = {
#     'result_sha' : val_result_sha,
#     'det_root' : val_det_root,
#     'seq_eval' : val_seq_eval if seq_eval_mode == 'all' else [seq_eval_mode],
#     'len_record' : val_len_record,
#     'detection_feature_path' : val_detection_feature_path,
#   }
#
#   training_config_dict = train_config_dict if training_split == 'train' else val_config_dict
#   evaluation_config_dict = val_config_dict if evaluation_split == 'val' else train_config_dict
#
#   return  training_config_dict, evaluation_config_dict, det_id2str, hw, data_root, cav_id_list, differentiable_kalman_filter_config, learning_rate, weight_decay
#
#
# def write_tensorboard(writer, freq, freq_idx, loss_types,
#   training_metrics_dict, training_loss_stats_avg, training_loss_stats_avg_dict,
#   evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict):
#   '''
#   Input
#     freq: one of ['epoch', 'step']
#     freq_idx: epoch index or step index
#   '''
#
#   if training_metrics_dict is not None:
#     writer.add_scalar('training/%s/sAMOTA' % freq, training_metrics_dict['sAMOTA'], freq_idx)
#     writer.add_scalar('training/%s/AMOTA' % freq, training_metrics_dict['AMOTA'], freq_idx)
#     writer.add_scalar('training/%s/AMOTP' % freq, training_metrics_dict['AMOTP'], freq_idx)
#
#   if training_loss_stats_avg is not None:
#     writer.add_scalar('training/%s/loss' % freq, training_loss_stats_avg, freq_idx)
#     for loss_type in loss_types:
#       writer.add_scalar('training/%s/%s_loss' % (freq, loss_type), training_loss_stats_avg_dict[loss_type], freq_idx)
#
#   if evaluation_metrics_dict is not None:
#     writer.add_scalar('evaluation/%s/sAMOTA' % freq, evaluation_metrics_dict['sAMOTA'], freq_idx)
#     writer.add_scalar('evaluation/%s/AMOTA' % freq, evaluation_metrics_dict['AMOTA'], freq_idx)
#     writer.add_scalar('evaluation/%s/AMOTP' % freq, evaluation_metrics_dict['AMOTP'], freq_idx)
#
#   if evaluation_loss_stats_avg is not None:
#     writer.add_scalar('evaluation/%s/loss' % freq, evaluation_loss_stats_avg, freq_idx)
#     for loss_type in loss_types:
#       writer.add_scalar('evaluation/%s/%s_loss' % (freq, loss_type), evaluation_loss_stats_avg_dict[loss_type], freq_idx)
#
#   # # =====================添加过程噪声损失记录===============================
#   # if 'process_noise' in loss_types:
#   #   if training_loss_stats_avg_dict and 'process_noise' in training_loss_stats_avg_dict:
#   #     writer.add_scalar('training/%s/process_noise_loss' % freq, training_loss_stats_avg_dict['process_noise'],
#   #                       freq_idx)
#   #
#   #   if evaluation_loss_stats_avg_dict and 'process_noise' in evaluation_loss_stats_avg_dict:
#   #     writer.add_scalar('evaluation/%s/process_noise_loss' % freq, evaluation_loss_stats_avg_dict['process_noise'],
#   #                       freq_idx)
#   #=====================添加trajectory_q_loss和rq_consistency损失记录===============================
#   if 'trajectory_q_loss' in loss_types:
#     if training_loss_stats_avg_dict and 'trajectory_q_loss' in training_loss_stats_avg_dict:
#       writer.add_scalar('training/%s/trajectory_q_loss' % freq,
#                         training_loss_stats_avg_dict['trajectory_q_loss'], freq_idx)
#
#     if evaluation_loss_stats_avg_dict and 'trajectory_q_loss' in evaluation_loss_stats_avg_dict:
#       writer.add_scalar('evaluation/%s/trajectory_q_loss' % freq,
#                         evaluation_loss_stats_avg_dict['trajectory_q_loss'], freq_idx)
#
#   if 'rq_consistency' in loss_types:
#     if training_loss_stats_avg_dict and 'rq_consistency' in training_loss_stats_avg_dict:
#       writer.add_scalar('training/%s/rq_consistency_loss' % freq,
#                         training_loss_stats_avg_dict['rq_consistency'], freq_idx)
#
#     if evaluation_loss_stats_avg_dict and 'rq_consistency' in evaluation_loss_stats_avg_dict:
#       writer.add_scalar('evaluation/%s/rq_consistency_loss' % freq,
#                         evaluation_loss_stats_avg_dict['rq_consistency'], freq_idx)
#
#   return
#
#
# def _build_q_config(differentiable_kalman_filter_config, q_learning_method):
#   """
#   从differentiable_kalman_filter_config构造q_learning_config
#
#   Args:
#       differentiable_kalman_filter_config: 原始卡尔曼滤波配置
#       q_learning_method: Q学习方法 ('innovation_based', 'physics_constrained', 'multiscale_attention')
#
#   Returns:
#       q_learning_config: Q学习配置字典
#   """
#   if differentiable_kalman_filter_config is not None:
#     # 从原配置中提取基础信息
#     dim_x = differentiable_kalman_filter_config.get('dim_x', 10)
#     dim_z = differentiable_kalman_filter_config.get('dim_z', 7)
#
#     # 获取观测协方差设置
#     obs_cov_setting = differentiable_kalman_filter_config.get('observation_covariance_setting', {})
#     feature_channel_size = obs_cov_setting.get('feature_channel_size', 512)
#     feature_region_size = obs_cov_setting.get('feature_region_size', 5)
#
#     # 构造Q学习配置
#     q_learning_config = {
#       # 基础维度信息
#       'dim_x': dim_x,
#       'dim_z': dim_z,
#
#       # Q学习方法
#       'q_learning_method': q_learning_method,
#
#       # 网络配置
#       'feature_channel_size': feature_channel_size,
#       'feature_region_size': feature_region_size,
#
#       # 学习参数
#       'learning_rate': 1e-4,
#       'history_length': 10,  # 历史序列长度
#       'hidden_dim': 128,  # 隐藏层维度
#
#       # 损失权重
#       'loss_weights': {
#         'innovation_consistency': 1.0,  # 创新一致性损失权重
#         'regularization': 0.1,  # 正则化损失权重
#         'positive_definiteness': 0.5,  # 正定性损失权重
#         'physics_constraint': 0.3,  # 物理约束损失权重（如果使用）
#         'attention_regularization': 0.2  # 注意力正则化权重（如果使用）
#       },
#
#       # 其他配置
#       'gt_data_association_threshold': differentiable_kalman_filter_config.get(
#         'gt_data_association_threshold', 3.0),
#       'dkf_type': differentiable_kalman_filter_config.get('dkf_type', 'multi_sensor'),
#
#       # 根据不同方法的特定配置
#       # **self._get_method_specific_config(q_learning_method, differentiable_kalman_filter_config)
#     }
#   else:
#     # 如果没有提供配置，使用默认值
#     q_learning_config = {
#       'dim_x': 10,
#       'dim_z': 7,
#       'q_learning_method': q_learning_method,
#       'feature_channel_size': 512,
#       'feature_region_size': 5,
#       'learning_rate': 1e-4,
#       'history_length': 10,
#       'hidden_dim': 128,
#       'loss_weights': {
#         'innovation_consistency': 1.0,
#         'regularization': 0.1,
#         'positive_definiteness': 0.5,
#         'physics_constraint': 0.3,
#         'attention_regularization': 0.2
#       },
#       'gt_data_association_threshold': 3.0,
#       'dkf_type': 'multi_sensor',
#       # **get_method_specific_config(q_learning_method, None)
#     }
#
#   return q_learning_config
#
# def  main_per_cat_multi_sensor_differentiable_kalman_filter(cfg, cat, log, ID_start, dtype, device, run_evaluation_every_epoch, seq_eval_mode, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis, num_frames_backprop, num_frames_per_sub_seq, num_epochs, load_model_path, save_dir_prefix, training_split, evaluation_split, feature, regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, clip_grad_norm, use_reversed_cav_id_list, # 新增参数
#                                                           enable_learnable_Q=False,
#                                                            q_learning_method='innovation_based',
#                                                            trajectory_q_loss_weight=10,  # 新增参数
#                                                            rq_consistency_loss_weight=0,
#                                                            q_net_type='velocity_adaptive'):
#
#   # get data-cat-split specific path
#   training_config_dict, evaluation_config_dict, det_id2str, hw, data_root, cav_id_list, differentiable_kalman_filter_config, learning_rate, weight_decay = get_config(cfg, cat, training_split, evaluation_split, seq_eval_mode, use_reversed_cav_id_list)
#   print('training_config_dict: ', training_config_dict)
#   print('evaluation_config_dict: ', evaluation_config_dict)
#
#   # load ground truth label
#   training_ground_truth_data_all = load_ground_truth_data_all(training_config_dict['result_sha'], seq_eval_mode, training_split)
#   evaluation_ground_truth_data_all = load_ground_truth_data_all(evaluation_config_dict['result_sha'], seq_eval_mode, evaluation_split)
#
#   # tensorboad setup
#   # write tensorboard file in the training folder epoch_idx=0
#   tensorboard_save_folder =  'training_' + training_config_dict['result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % 0
#   tensorboard_save_folder = os.path.join(save_dir_prefix, tensorboard_save_folder)
#   tensorboard_save_dir = os.path.join(cfg.save_root, tensorboard_save_folder, 'tensorboard'); mkdir_if_missing(tensorboard_save_dir)
#   print('tensorboard_save_dir: ', tensorboard_save_dir)
#   writer = SummaryWriter(tensorboard_save_dir)
#
#   # model
#   torch.autograd.set_detect_anomaly(True)
#   observation_covariance_net_dict = {}
#   traj_based_q_net_dict = {}
#   q_net_dict = {}
#   if q_net_type == 'fixed':
#     # 固定Q，不使用学习
#     enable_learnable_Q = False
#     q_net_dict = None
#     print("\n使用固定Q矩阵（baseline）")
#   else:
#     # 使用可学习Q
#     enable_learnable_Q = True
#
#     for cav_id in cav_id_list:
#       q_net, q_loss_fn = create_q_net(
#         q_net_type=q_net_type,
#         config=q_learning_config,
#         device=device,
#         dtype=dtype
#       )
#       q_net_dict[cav_id] = q_net.to(device)
#
#       if load_model_path != '':
#         q_net_model_file = load_model_path.replace('model_', f'{q_net_type}_q_net_model_').replace('ego', cav_id)
#         if os.path.exists(q_net_model_file):
#           print(f'Loading Q-net weights for {cav_id}: {q_net_model_file}')
#           q_net_dict[cav_id].load_state_dict(torch.load(q_net_model_file, map_location=device))
#
#   # innovation_based_q_net = {}  # 新增
#   for cav_id in cav_id_list:
#     observation_covariance_net_dict[cav_id] = ObservationCovarianceNet(differentiable_kalman_filter_config, feature).to(device)
#     q_learning_config = _build_q_config(differentiable_kalman_filter_config, q_learning_method)
#     traj_based_q_net_dict[cav_id] = TrajectoryBasedQNet(q_learning_config, device, dtype).to(device)
#
#     # ================过程噪声网络 (可以选择每个CAV单独一个或共享一个)======================
#     # if enable_learnable_Q:
#     #   if use_multiple_nets:
#     #     # 每个CAV独立的过程噪声网络
#     #     process_noise_net_dict[cav_id] = ProcessNoiseNet(
#     #       differentiable_kalman_filter_config=differentiable_kalman_filter_config,
#     #       device=device,
#     #       dtype=dtype,
#     #       history_length=10,
#     #       hidden_dim=128
#     #     ).to(device)
#     #   else:
#     #     # 共享过程噪声网络
#     #     if 'shared' not in process_noise_net_dict:
#     #       process_noise_net_dict['shared'] = ProcessNoiseNet(
#     #         differentiable_kalman_filter_config=differentiable_kalman_filter_config,
#     #         device=device,
#     #         dtype=dtype,
#     #         history_length=10,
#     #         hidden_dim=128
#     #       ).to(device)
#
#     if load_model_path != '':
#       model_file = load_model_path.replace('ego', cav_id)
#       print('model_file: ', model_file)
#       observation_covariance_net_dict[cav_id].load_state_dict(torch.load(model_file, map_location=device))
#       #=======================冻结权重==============================
#
#       # # 冻结BEV特征提取部分
#       # for param in observation_covariance_net_dict[cav_id].bev_conv_and_max_pool.parameters():
#       #   param.requires_grad = False
#       #
#       # # 冻结位置编码部分
#       # for param in observation_covariance_net_dict[cav_id].positional_encoding_linear.parameters():
#       #   param.requires_grad = False
#       #
#       #   print(f"   已冻结 {cav_id} 网络的BEV和位置编码层")
#       #====================== 尝试加载过程噪声网络权重=====================
#       if enable_learnable_Q:
#         traj_based_q_net_model_file = model_file.replace('model_', 'traj_based_q_net_model_')
#         if os.path.exists(traj_based_q_net_model_file):
#           print('Loading process noise model file: ', traj_based_q_net_model_file)
#           traj_based_q_net_dict[cav_id].load_state_dict(torch.load(traj_based_q_net_model_file))
#           # if use_multiple_nets:
#           #   process_noise_net_dict[cav_id].load_state_dict(torch.load(process_noise_model_file))
#           # else:
#           #   process_noise_net_dict['shared'].load_state_dict(torch.load(process_noise_model_file))
#       # MY_DEBUG
#       if False:
#         for param in observation_covariance_net_dict[cav_id].parameters():
#           print(torch.mean(param.data))
#           print(torch.min(param.data)) # 1e-6
#           print(torch.max(param.data)) # 124
#           print(torch.norm(param.data))
#           assert False
#
#   # optimizer
#   parameters_list = list(observation_covariance_net_dict['ego'].parameters())
#   if use_multiple_nets and '1' in cav_id_list:
#     parameters_list = parameters_list + list(observation_covariance_net_dict['1'].parameters())
#
#   # =============================添加traj_based_q_net网络参数===============================
#   if enable_learnable_Q:
#     parameters_list = list(traj_based_q_net_dict['ego'].parameters())
#     if use_multiple_nets and '1' in cav_id_list:
#     # parameters_list = parameters_list + list(innovation_based_q_net.parameters())
#       parameters_list = parameters_list + list(traj_based_q_net_dict['1'].parameters())
#     # if use_multiple_nets:
#     #   for cav_id in cav_id_list:
#     #     if cav_id in innovation_based_q_net:
#     #       parameters_list = parameters_list + list(innovation_based_q_net.parameters())
#     # else:
#     #   if 'shared' in innovation_based_q_net:
#     #     parameters_list = parameters_list + list(innovation_based_q_net.parameters())
#
#   optimizer = torch.optim.Adam(parameters_list, lr=learning_rate, weight_decay=weight_decay)
#   loss_types = ['regression', 'association', 'det_neg_log_likelihood']
#   if enable_learnable_Q:
#     # loss_types.append('process_noise')
#     loss_types.extend(['trajectory_q_loss', 'trajectory_s_loss'])
#
#   mail = mailpy.Mail('')
#   threshold_3D_IOU = 0.25
#   # initial evaluation before training loop
#   # MY_DEBUG
#   run_init_evaluation = True
#   if run_init_evaluation:
#     print('Run initial evaluation before training loop')
#     # let epoch_idx = 0
#     # start = time.time()
#     evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict = track_and_evaluate(save_dir_prefix, cav_id_list, observation_covariance_net_dict,
#                                                                                                             evaluation_ground_truth_data_all, evaluation_config_dict, evaluation_split,
#                                                                                                             cfg, seq_eval_mode, 0, cat, ID_start, hw, log, dtype, device,
#                                                                                                             differentiable_kalman_filter_config, det_id2str, force_gt_as_predicted_track,
#                                                                                                             use_static_default_R, use_multiple_nets, show_vis, save_vis, mail, threshold_3D_IOU,
#                                                                                                             loss_types, writer, regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight
#                                                                                                             ,q_learning_config, enable_learnable_Q=enable_learnable_Q, q_net_dict=traj_based_q_net_dict, trajectory_q_loss_weight=trajectory_q_loss_weight, rq_consistency_loss_weight=rq_consistency_loss_weight)
#
#     # end = time.time()
#     # print("并行评估时间：", end - start)
#     print('evaluation_metrics_dict: ', evaluation_metrics_dict)
#     # write tensorboard for this epoch
#     write_tensorboard(writer, 'epoch', 0, loss_types,
#       None, None, None,
#       evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict)
#
#
#   # number of optimizer.step() called
#   step_index = 0
# #===============训练==================================================================================================================================
#   # training loop start with epoch_idx = 1
#   for epoch_idx in range(1, num_epochs+1):
#     print('epoch_idx: ', epoch_idx)
#
#     training_save_folder = 'training_' + training_config_dict['result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % epoch_idx
#     training_save_folder = os.path.join(save_dir_prefix, training_save_folder)
#     training_save_dir = os.path.join(cfg.save_root, training_save_folder); mkdir_if_missing(training_save_dir)
#     #print('training_save_dir: ', training_save_dir)
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_all_H1_epoch_0
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_0001_H1_epoch_0
#
#     # create eval dir for each hypothesis
#     eval_dir_dict = dict()
#     for index in range(cfg.num_hypo):
#       eval_dir_dict[index] = os.path.join(training_save_dir, 'data_%d' % index); mkdir_if_missing(eval_dir_dict[index])
#     #print('eval_dir_dict[0]: ', eval_dir_dict[0])
#     # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0/data_0
#
#     # save tracking results every epoch
#     save_tracking_results = True
#     training_loss_stats_avg, training_loss_stats_avg_dict, step_index = train_one_epoch(
#       True, loss_types, writer, step_index,
#       save_tracking_results,
#       training_config_dict['seq_eval'], cav_id_list,
#       training_config_dict['det_root'], eval_dir_dict, training_save_dir, training_config_dict['result_sha'],
#       cfg, cat, ID_start, hw, log,
#       dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict,
#       optimizer, training_config_dict['len_record'], training_config_dict['detection_feature_path'],
#       training_ground_truth_data_all, det_id2str,
#       num_frames_backprop, num_frames_per_sub_seq, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis, training_split,
#       regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, clip_grad_norm, parameters_list, q_learning_config,
#     enable_learnable_Q=enable_learnable_Q,q_net_dict=traj_based_q_net_dict,    trajectory_q_loss_weight=0.1,  # 新增参数
#     rq_consistency_loss_weight=0.1)
#
#     # save model checkpoint
#     for cav_id in cav_id_list:
#       model_save_file = os.path.join(training_save_dir, 'model_%s_epoch_%d.pth' % (cav_id, epoch_idx))
#       torch.save(observation_covariance_net_dict[cav_id].state_dict(), model_save_file)
#
#     # 添加保存过程噪声网络的代码
#     if enable_learnable_Q:
#         # for key, net in process_noise_net_dict.items():
#       # process_noise_model_save_file = os.path.join(training_save_dir, 'process_noise_model_%s_epoch_%d.pth' % (key, epoch_idx))
#       traj_based_q_net_model_save_file = os.path.join(training_save_dir, 'traj_based_q_net_model_epoch_%d.pth' % epoch_idx)
#       torch.save(traj_based_q_net_dict[cav_id].state_dict(), traj_based_q_net_model_save_file)
#     print("保存完成")
#     # tracer.stop()
#     # tracer.save()
# # # =====================保存CIdeep权重==========================================================================
# #     CI_trainning_save_dir = '/mnt/d/pycharm_projects/DMSTrack-master/YJQTrack/official_models_ci'
# #     os.makedirs(CI_trainning_save_dir, exist_ok=True)  # 确保目录存在
# #     ci_model_save_file = os.path.join(CI_trainning_save_dir, 'ci_weight_predictor_epoch_%d.pth' % epoch_idx)
# #     CovarianceIntersection.save_weights(ci_model_save_file)
# # # ============================================================================================================
#
#     # run evaluation command
#     if run_evaluation_every_epoch:
#
#       # usually we do not run evaluation on training split
#       # enable it only for debug
#       run_evaluation_on_training_split = False
#       if run_evaluation_on_training_split:
#         # 'all', '0000', '0001', ...
#         # this command will evaluate the current training split's tracking results
#         #training_split_evaluation_command = 'python3 scripts/KITTI/evaluate.py ' + training_save_folder + ' 1 3D 0.25 %s %s' % (seq_eval_mode, training_split)
#         #print('training_split_evaluation_command: ', training_split_evaluation_command)
#         #returned_value = os.system(training_split_evaluation_command)
#         #print('returned_value: ', returned_value)
#
#         # instead of running another python command
#         # call ab3dmot script evaluation function directly
#         training_metrics_dict = evaluate(training_save_folder, mail, 1, True, False, threshold_3D_IOU, True, seq_eval_mode, training_split)
#         print('training_metrics_dict: ', training_metrics_dict)
#       else:
#         training_metrics_dict = None
#
#
#       # Evalue on evaluation split
#       # if we want to evaluate on the evaluation split, need to generate the tracing results of val set and run eval command
#       # inside the following evaluate() function, we will also save model checkpoints
#       evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict = track_and_evaluate(save_dir_prefix, cav_id_list, observation_covariance_net_dict, evaluation_ground_truth_data_all, evaluation_config_dict, evaluation_split, cfg, seq_eval_mode, epoch_idx, cat, ID_start, hw, log, dtype, device, differentiable_kalman_filter_config, det_id2str, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis, mail, threshold_3D_IOU, loss_types, writer,
#       regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight)
#       print('evaluation_metrics_dict: ', evaluation_metrics_dict)
#
#       # write tensorboard for this epoch
#       write_tensorboard(writer, 'epoch', epoch_idx, loss_types,
#         training_metrics_dict, training_loss_stats_avg, training_loss_stats_avg_dict,
#         evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict)
#
#
#   return ID_start
#
#
# def main(args):
#   torch.manual_seed(0)
#   np.random.seed(0)
#
#   # load config files
#   config_path = './configs/%s.yml' % args.dataset
#   cfg, settings_show = Config(config_path)
#
#   # overwrite split and detection method
#   if args.split is not '': cfg.split = args.split
#   if args.det_name is not '': cfg.det_name = args.det_name
#
#   # print configs
#   time_str = get_timestring()
#   log = os.path.join(cfg.save_root, args.save_dir_prefix, 'log/log_%s_%s_%s.txt' % (time_str, cfg.dataset, cfg.split))
#   mkdir_if_missing(log); log = open(log, 'w')
#   print_log(args, log)
#   for idx, data in enumerate(settings_show):
#     print_log(data, log, display=False)
#
#   print('config_path: ', config_path)
#   print('cfg.save_root: ', cfg.save_root)
#   print('args.save_dir_prefix: ', args.save_dir_prefix)
#
#   # global ID counter used for all categories, not start from 1 for each category to prevent different
#   # categories of objects have the same ID. This allows visualization of all object categories together
#   # without ID conflicting, Also use 1 (not 0) as start because MOT benchmark requires positive ID
#   ID_start = 1
#
#
#   # deep learning parameter
#   dtype = torch.float32
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   print('device: ', device)
#
#   # run tracking for each category
#   for cat in cfg.cat_list:
#     ID_start = main_per_cat_multi_sensor_differentiable_kalman_filter(cfg, cat, log, ID_start, dtype, device,
#       args.run_evaluation_every_epoch,
#       args.seq_eval_mode,
#       args.force_gt_as_predicted_track,
#       args.use_static_default_R,
#       args.use_multiple_nets,
#       args.show_vis,
#       args.save_vis,
#       args.num_frames_backprop,
#       args.num_frames_per_sub_seq,
#       args.num_epochs,
#       args.load_model_path,
#       args.save_dir_prefix,
#       args.training_split,
#       args.evaluation_split,
#       args.feature,
#       args.regression_loss_weight,
#       args.association_loss_weight,
#       args.det_neg_log_likelihood_loss_weight,
#       args.clip_grad_norm,
#       args.use_reversed_cav_id_list)
#
#   # combine results for every category
#   print_log('\ncombining results......', log=log)
#
#   # MY_COMMENT: combine_trk_cat is not required since we only track 'Car'
#   #combine_trk_cat(cfg.split, cfg.dataset, cfg.det_name, 'H%d' % cfg.num_hypo, cfg.num_hypo)
#   #for epoch_idx in range(num_epochs):
#   #  combine_trk_cat(cfg.split, cfg.dataset, cfg.det_name, 'H%d_epoch_%d' % (cfg.num_hypo, epoch_idx), cfg.num_hypo)
#
#   print_log('\nDone!', log=log)
#   log.close()
#
# if __name__ == '__main__':
#   args = parse_args()
#   tracer = VizTracer()
#   # tracer.start()
#   main(args)


from __future__ import print_function
import matplotlib;

matplotlib.use('Agg')
import os, numpy as np, time, sys, argparse
# sys.path.append('/ocean/projects/cis230055p/hchiu1/my_cooperative_tracking/AB3DMOT')
# sys.path.append('/ocean/projects/cis230055p/hchiu1/my_cooperative_tracking/AB3DMOT/Xinshuo_PyToolbox')
# sys.path.append('/ocean/projects/cis230055p/hchiu1/my_cooperative_tracking/V2V4Real')
from AB3DMOT.AB3DMOT_libs.utils import Config, get_subfolder_seq
from AB3DMOT.AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_results, save_affinity
from AB3DMOT.scripts.KITTI.evaluate import trackingEvaluation, evaluate
from AB3DMOT.scripts.post_processing.combine_trk_cat import combine_trk_cat
from AB3DMOT.scripts.KITTI import mailpy
from AB3DMOT.scripts.KITTI import munkres
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing, save_txt_file
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import get_timestring, print_log

import torch
from DMSTrack.model_delay import ObservationCovarianceNet, DMSTrack
from YJQTrack.q_noise_enhanced.enhanced_dmstrack import EnhancedDMSTrack
from YJQTrack.q_noise_enhanced.full_q_for_ral_r1.model_dkf_delay import BestForNowTrack
from YJQTrack.q_noise_enhanced.parallel_enhanced_dmstrack import ParallelEnhancedDMSTrack
from YJQTrack.visualization import visualize
from tensorboardX import SummaryWriter
from YJQTrack.CI.q_noise_net import ProcessNoiseNet
from YJQTrack.q_noise_enhanced.q_noise_with_innovation import InnovationBasedQNet
from YJQTrack.q_noise_enhanced.full_q_for_ral_r1.q_noise_with_trajectory import TrajectoryBasedQNet
import os

from YJQTrack.CI.covariance_intersection_deep import CovarianceIntersection
# from viztracer import VizTracer
from YJQTrack.q_noise_enhanced.serial_dmstrack_with_ci import SeriallDMSTrackWithCI
# from YJQTrack.q_noise_enhanced.best_for_now import BestForNowTrackV1
# from viztracer import VizTracer
import torch.nn as nn


def parse_args():
  parser = argparse.ArgumentParser(description='DMSTrack')
  parser.add_argument('--dataset', type=str, default='v2v4real', help='v2v4real')
  parser.add_argument('--split', type=str, default='', help='train, val, test')
  parser.add_argument('--det_name', type=str, default='', help='multi_sensor_differentiable_kalman_filter')
  parser.add_argument('--seq_eval_mode', type=str, default='all', help='which seq to eval: all, 0000, 0001, ...')

  parser.add_argument('--run_evaluation_every_epoch', action='store_true', help='auto run evaluation every epoch')
  parser.add_argument('--force_gt_as_predicted_track', action='store_true',
                      help='during training force gt as predicted track before data association and update')
  parser.add_argument('--use_static_default_R', action='store_true',
                      help='use static default observation noise covariance')
  parser.add_argument('--use_multiple_nets', action='store_true',
                      help='use multiple observation covariance nets for multiple cars')
  parser.add_argument('--num_frames_backprop', type=int, default=-1,
                      help='number of past frames that the loss backpropagates. If -1, loss is calculated for the full (sub) sequence.')
  parser.add_argument('--num_frames_per_sub_seq', type=int, default=-1,
                      help='number of frames per sub seq during training data chunk and random shuffle. If -1, each sequence is a sub sequence.')
  parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
  parser.add_argument('--load_model_path', type=str, default='', help='path to ego model checkpoint')
  parser.add_argument('--save_dir_prefix', type=str, default='', help='prefix of save dir')
  parser.add_argument('--feature', type=str, default='fusion', help='model feature, fusion or bev or pos')
  parser.add_argument('--regression_loss_weight', type=float, default=0, help='regression loss weight')
  parser.add_argument('--association_loss_weight', type=float, default=0, help='association loss weight')
  parser.add_argument('--det_neg_log_likelihood_loss_weight', type=float, default=0,
                      help='detection negative log likelihood loss weight')
  parser.add_argument('--clip_grad_norm', type=float, default=-1,
                      help='clip gradient norm up to this value. -1 means no clip')
  parser.add_argument('--use_reversed_cav_id_list', action='store_true', help='process cav_1 then ego during kf update')

  parser.add_argument('--training_split', type=str, default='train', help='data split during training')
  parser.add_argument('--evaluation_split', type=str, default='val', help='data split during evaluation')

  # visualization
  parser.add_argument('--show_vis', action='store_true',
                      help='show open3d interactive visualization result')
  parser.add_argument('--save_vis', action='store_true',
                      help='save open3d image and video visualization result')
  parser.add_argument('--delay_config', type=str, default='ego:0,1:1',
                      help='Delay configuration: ego:0,1:2 means ego has 0 delay, cav_1 has 2 frames delay')

  args = parser.parse_args()
  return args

def parse_delay_config(delay_str):
    """
    解析延迟配置字符串
    'ego:0,1:2' -> {'ego': 0, '1': 2}
    """
    delay_dict = {}
    for item in delay_str.split(','):
        cav_id, delay = item.split(':')
        delay_dict[cav_id.strip()] = int(delay)
    return delay_dict

def get_global_timestamp_index(seq_name, frame, len_record):
  # only for v2v4real no_fusion_keep_all
  seq_id = int(seq_name)
  # print('seq_id: ', seq_id)
  # print('frame: ', frame)
  # print('len_record: ', len_record)
  # test set: [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
  # print('detection_feature_path: ', detection_feature_path)
  # /home/eddy/my_cooperative_tracking/V2V4Real/official_models/no_fusion_keep_all/npy/

  if seq_id == 0:
    start_global_timestamp_index = 0
  else:
    start_global_timestamp_index = len_record[seq_id - 1]

  global_timestamp_index = start_global_timestamp_index + frame

  return global_timestamp_index


def load_transformation_matrix(global_timestamp_index, detection_feature_path, cav_id):
  transformation_matrix_file = os.path.join(detection_feature_path, cav_id,
                                            '%04d_transformation_matrix.npy' % global_timestamp_index)
  transformation_matrix = np.load(transformation_matrix_file)
  # print('transformation_matrix.shape: ', transformation_matrix.shape)
  # (4, 4)
  # assert False
  return transformation_matrix


def load_detection_feature_cobevt_single_sensor(global_timestamp_index, detection_feature_path, cav_id):
  detection_feature_file = os.path.join(detection_feature_path, cav_id, '%04d_feature.npy' % global_timestamp_index)
  # print('detection_feature_file: ', detection_feature_file)
  detection_feature = np.load(detection_feature_file)
  # print('detection_feature.shape: ', detection_feature.shape)
  # (5, 256, 5, 5)
  # print('np.sum(detection_feature[0], axis=0): ', np.sum(detection_feature[0], axis=0))

  early_detection_feature_file = os.path.join(detection_feature_path, cav_id,
                                              '%04d_early_feature.npy' % global_timestamp_index)
  early_detection_feature = np.load(early_detection_feature_file)
  # print('early_detection_feature.shape: ', early_detection_feature.shape)
  # (5, 256, 5, 5)

  # concate the two feature
  aggregated_detection_feature = np.concatenate([detection_feature, early_detection_feature], axis=1)
  # print('aggregated_detection_feature.shape: ', aggregated_detection_feature.shape)
  # (5, 512, 5, 5)

  return aggregated_detection_feature


def load_detection_feature(global_timestamp_index, detection_feature_path, cav_id):
  detection_feature_file = os.path.join(detection_feature_path, cav_id, '%04d_feature.npy' % global_timestamp_index)
  # print('detection_feature_file: ', detection_feature_file)
  detection_feature = np.load(detection_feature_file)
  # print('detection_feature.shape: ', detection_feature.shape)
  # (5, 256, 5, 5)
  # print('np.sum(detection_feature[0], axis=0): ', np.sum(detection_feature[0], axis=0))

  # repeat (1, 1, 4, 4)
  detection_feature = np.repeat(detection_feature, 4, axis=2)
  detection_feature = np.repeat(detection_feature, 4, axis=3)
  # print('detection_feature.shape: ', detection_feature.shape)
  # (5, 256, 20, 20)
  # print('np.sum(detection_feature[0], axis=0): ', np.sum(detection_feature[0], axis=0))

  early_detection_feature_file = os.path.join(detection_feature_path, cav_id,
                                              '%04d_early_feature.npy' % global_timestamp_index)
  early_detection_feature = np.load(early_detection_feature_file)
  # print('early_detection_feature.shape: ', early_detection_feature.shape)
  # (5, 64, 20, 20)

  # concate the two feature
  aggregated_detection_feature = np.concatenate([detection_feature, early_detection_feature], axis=1)
  # print('aggregated_detection_feature.shape: ', aggregated_detection_feature.shape)
  # (5, 320, 20, 20)

  return aggregated_detection_feature


def load_ground_truth_data_all(result_sha, seq_eval_mode, split):
  e = trackingEvaluation(result_sha, evaluate_v2v4real=True, seq_eval_mode=seq_eval_mode, v2v4real_split=split)
  e.loadGroundtruth()

  # ground_truth = e.groundtruth
  # print('e.n_gt_seq: ', e.n_gt_seq) # [6, 12, 7, 73, 65, 137, 56, 32, 27]
  # print('e.n_gt_trajectories: ', e.n_gt_trajectories) # 415
  # print('len(ground_truth): ', len(ground_truth)) # 9 sequences
  # print('len(ground_truth[0]): ', len(ground_truth[0])) # 147 frames in the first sequence
  # print('len(ground_truth[0][0]): ', len(ground_truth[0][0])) # 1
  # print('ground_truth[0][0][0]: ', ground_truth[0][0][0])
  # print('ground_truth[0][1][0]: ', ground_truth[0][1][0])

  ground_truth_data_all = e.groundtruth
  return ground_truth_data_all


def get_ground_truth_data_single_frame(seq_idx_in_eval, frame, ground_truth_data_all, dtype, device):
  '''
  Input:
    ground_truth_data_all:
      list of seq
        list of frame
          list of object
            object: class tData in AB3DMOT/scripts/KITTI/evaluate.py
  Output:
    N: number of ground-truth objects in this frame
    gt_boxes: tensor (N, 7):
      same order of kalman filter state: [x, y, z, theta, l, w, h]
    gt_ids: tensor (N)
  '''
  ground_truth_data_single_frame = ground_truth_data_all[seq_idx_in_eval][frame]
  gt_boxes = []
  gt_ids = []

  for i in range(len(ground_truth_data_single_frame)):
    object = ground_truth_data_single_frame[i]
    # print(object)
    gt_boxes.append(
      np.array(
        [object.x, object.y, object.z, object.ry, object.l, object.w, object.h]
      )
    )
    gt_ids.append(object.track_id)

  gt_boxes = np.stack(gt_boxes)
  gt_ids = np.stack(gt_ids)

  gt_boxes = torch.tensor(gt_boxes, dtype=dtype, device=device)
  gt_ids = torch.tensor(gt_ids, dtype=dtype, device=device)

  # print('gt_boxes: ', gt_boxes)
  # print('gt_ids: ', gt_ids)

  # print('gt_boxes.shape: ', gt_boxes.shape)
  # print('gt_ids.shape: ', gt_ids.shape)

  return gt_boxes, gt_ids


# Similar to AB3DMOT_libs/utils.py initialize()
def initialize_DMSTrack(cfg, seq_name, cat, ID_start, hw, log_file, dtype, device, differentiable_kalman_filter_config,
                        observation_covariance_net_dict, force_gt_as_predicted_track, use_static_default_R,
                        use_multiple_nets, split,delay_config,
                        q_learning_config, enable_learnable_Q=False, q_net_dict=None):
  # initialize the tracker and provide all path of data needed
  # 历史管理器配置
  history_manager_config = {
    'max_history_length': 20,
    'save_dir': None,  # 可以设置保存目录
    'save_interval': 100,
    'enable_export': False  # 训练时可以关闭导出
  }
  # initiate the tracker
  # tracker = DMSTrack(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start,
  #                    dtype=dtype, device=device,
  #                    differentiable_kalman_filter_config=differentiable_kalman_filter_config,
  #                    observation_covariance_net_dict=observation_covariance_net_dict,
  #                    force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R)

  # tracker = BestForNowTrackV1(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,
  #                     enable_learnable_Q=enable_learnable_Q)
  # tracker = EnhancedDMSTrack(cfg, cat, q_learning_config, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None,train=True,innovation_based_q_net=innovation_based_q_net)
  tracker = BestForNowTrack(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file,
                            ID_init=ID_start, dtype=dtype, device=device,
                            differentiable_kalman_filter_config=differentiable_kalman_filter_config,
                            observation_covariance_net_dict=observation_covariance_net_dict,
                            force_gt_as_predicted_track=force_gt_as_predicted_track,
                            use_static_default_R=use_static_default_R, enable_learnable_Q=enable_learnable_Q,delay_config=delay_config,
                            q_learning_config=q_learning_config, q_net_dict=q_net_dict)
  # tracker = ParallelEnhancedDMSTrack(cfg, cat, q_learning_config, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None,train=False,innovation_based_q_net=innovation_based_q_net,enable_learnable_Q=enable_learnable_Q)
  # tracker = SeriallDMSTrackWithCI(cfg, cat, q_learning_config, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None,train=False,innovation_based_q_net=innovation_based_q_net,enable_learnable_Q=enable_learnable_Q)
  # ====================设置过程噪声网络=======================
  # if enable_learnable_Q and q_net:
  #   tracker.process_noise_net = q_net
  # if use_multiple_nets:
  #   # 这里需要决定使用哪个CAV的网络，可以使用ego的
  #   if 'ego' in innovation_based_q_net:
  #     tracker.process_noise_net = innovation_based_q_net['ego']
  # else:
  #   if 'shared' in innovation_based_q_net:
  #     tracker.process_noise_net = innovation_based_q_net['shared']
  # tracker = DMSTrackWithHistory(cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=hw, log=log_file, ID_init=ID_start, dtype=dtype, device=device, differentiable_kalman_filter_config=differentiable_kalman_filter_config, observation_covariance_net_dict=observation_covariance_net_dict, force_gt_as_predicted_track=force_gt_as_predicted_track, use_static_default_R=use_static_default_R,history_manager_config=None)
  # MY_CODE:
  # seq_name is one of ['0000', '0001', ... '0008']
  # use it to read tracking label data to get the frames
  # ~/my_cooperative_tracking/AB3DMOT/scripts/KITTI/v2v4real_label/0000.txt
  tracking_label_file = '../AB3DMOT/scripts/KITTI/v2v4real_%s_label/%s.txt' % (split, seq_name)
  with open(tracking_label_file, 'r') as f:
    for line in f:
      frame_id = int(line.split(' ')[0])
  frame_list = list(range(frame_id + 1))
  # print('frame_list: ', frame_list)

  return tracker, frame_list


def get_len_per_seq_eval(len_record, seq_eval):
  '''
  Input
    len_record: all accumulated length of seqs
    seq_eval: list of seq_name being considered: ['0003'] or ['0000', '0001', ...]
  Output
    len_per_seq_eval: length of seq in seq_eval: val split example [198] or [147, 114, 144, ...]
  '''
  len_per_seq = [len_record[0]]
  for i in range(1, len(len_record)):
    len_per_seq.append(len_record[i] - len_record[i - 1])
  # print('len_record: ', len_record)
  # print('len_per_seq: ', len_per_seq)
  if len(seq_eval) == 1:
    len_per_seq_eval = [len_per_seq[int(seq_eval[0])]]
  else:
    len_per_seq_eval = len_per_seq
  # print('len_per_seq_eval: ', len_per_seq_eval)
  return len_per_seq_eval


def get_sub_seq_list_info(is_training, num_frames_per_sub_seq, len_record, len_per_seq_eval, seq_eval):
  '''
  Divide all sequences of a epoch into list of sub sequences.
  If is_training is True, randome shuffle and each sub seq has length num_frames_per_sub_seq.
  if num_frames_per_sub_seq == -1 or is_training is False,
  No random shuffle, each sequence is a sub sequence, list length is number of sequences

  Input:
    is_training
    num_frames_per_sub_seq : length of each sub seq if is_training is True
    len_record: all accumulated length of seqs: [147, 114, 144, 198, 180, 310, 304, 221, 375]
    len_per_seq_eval: length per seq in seq_eval: [198] or [147, 114, 144, 198, ...]
    seq_eval: list of seq_name being considered: ['0003'] or ['0000', '0001', ...]

  Output
    sub_seq_list_info: numpy [num_sub_seqs, 3]
      3:
        seq_name_list: global seq name
        seq_idx_in_eval_list: seq idx in seq_eval
        start_frame_idx_list: start frame idx in a seq
        end_frame_idx_list: end frame idx in a seq
  '''
  # print('num_frames_per_sub_seq: ', num_frames_per_sub_seq)
  # print('seq_eval: ', seq_eval)
  # print('len_record: ', len_record)
  # print('len_per_seq_eval: ', len_per_seq_eval)

  # not used
  # seq_id_from_seq_eval_name = [int(id) for id in seq_eval]
  # print('seq_id_from_seq_eval_name: ', seq_id_from_seq_eval_name)

  num_seqs_eval = len(seq_eval)

  if not is_training or num_frames_per_sub_seq == -1:
    num_sub_seqs = num_seqs_eval
    seq_name_list = seq_eval
    seq_idx_in_eval_list = list(range(num_seqs_eval))
    start_frame_idx_list = [0] * num_seqs_eval
    end_frame_idx_list = [length - 1 for length in len_per_seq_eval]
  else:
    seq_name_list = []
    seq_idx_in_eval_list = []
    start_frame_idx_list = []

    overlapping_sliding_window = False
    if overlapping_sliding_window:
      for i in range(len(seq_eval)):
        seq_name_list += [seq_eval[i]] * (len_per_seq_eval[i] - (num_frames_per_sub_seq - 1))
        seq_idx_in_eval_list += [i] * (len_per_seq_eval[i] - (num_frames_per_sub_seq - 1))
        start_frame_idx_list += list(range(len_per_seq_eval[i] - (num_frames_per_sub_seq - 1)))
      end_frame_idx_list = [start + num_frames_per_sub_seq - 1 for start in start_frame_idx_list]
    else:
      for i in range(len(seq_eval)):
        print('len_per_seq_eval[i]: ', len_per_seq_eval[i])
        print('num_frames_per_sub_seq: ', num_frames_per_sub_seq)
        print('len_per_seq_eval[i] // num_frames_per_sub_seq: ', len_per_seq_eval[i] // num_frames_per_sub_seq)
        num_sub_seqs_in_this_seq = len_per_seq_eval[i] // num_frames_per_sub_seq
        seq_name_list += [seq_eval[i]] * num_sub_seqs_in_this_seq
        seq_idx_in_eval_list += [i] * num_sub_seqs_in_this_seq
        start_frame_idx_list += [j * num_frames_per_sub_seq for j in range(num_sub_seqs_in_this_seq)]
      end_frame_idx_list = [start + num_frames_per_sub_seq - 1 for start in start_frame_idx_list]

    assert (len(seq_name_list) == len(start_frame_idx_list))
    assert (len(seq_idx_in_eval_list) == len(start_frame_idx_list))

  print('seq_name_list: ', seq_name_list)
  print('seq_idx_in_eval_list: ', seq_idx_in_eval_list)
  print('start_frame_idx_list: ', start_frame_idx_list)
  print('end_frame_idx_list: ', end_frame_idx_list)

  return seq_name_list, seq_idx_in_eval_list, start_frame_idx_list, end_frame_idx_list


def save_frame_tracking_results(save_dir, seq_name, frame_data_list):
  '''
  保存单个序列的所有帧数据

  frame_data_list: list of dict, 每个dict包含一帧的数据:
      {
          'frame': int,
          'tracking_results': np.array,
          'gt_boxes': np.array,
          'gt_ids': np.array,
          'track_P': list
      }
  '''
  import json

  # JSON格式（易读）
  json_file = os.path.join(save_dir, f'seq_{seq_name}_summary.json')
  summary = {
    'sequence': seq_name,
    'total_frames': len(frame_data_list),
    'frames': []
  }

  for data in frame_data_list:
    frame_info = {
      'frame_id': int(data['frame']),
      'num_tracks': len(data['tracking_results']),
      'num_gt': len(data['gt_boxes']),
      'tracks': [],
      'ground_truth': []
    }

    # 跟踪结果
    for trk in data['tracking_results']:
      frame_info['tracks'].append({
        'id': int(trk[7]),
        'position': {'x': float(trk[3]), 'y': float(trk[4]), 'z': float(trk[5])},
        'size': {'h': float(trk[0]), 'w': float(trk[1]), 'l': float(trk[2])},
        'rotation': float(trk[6]),
        'score': float(trk[14])
      })

    # Ground truth
    for gt_box, gt_id in zip(data['gt_boxes'], data['gt_ids']):
      frame_info['ground_truth'].append({
        'id': int(gt_id),
        'position': {'x': float(gt_box[0]), 'y': float(gt_box[1]), 'z': float(gt_box[2])},
        'size': {'h': float(gt_box[6]), 'w': float(gt_box[5]), 'l': float(gt_box[4])},
        'rotation': float(gt_box[3])
      })

    summary['frames'].append(frame_info)

  with open(json_file, 'w') as f:
    json.dump(summary, f, indent=2)

  print(f"[Info] Saved summary to {json_file}")


# 在 train_one_epoch 函数中
def compute_adaptive_loss_weights(loss_avg_dict, base_weights, dtype, device, max_adjustment=1000.0):
  """
  动态调整权重，使各损失贡献相当

  注意：loss_avg_dict 中的值应该是已经计算好的平均值（标量张量）
  """
  scales = {}
  for loss_type in ['regression', 'association', 'det_neg_log_likelihood',
                    'trajectory_q_loss', 'trajectory_s_loss']:
    if loss_type in loss_avg_dict:
      loss_val = loss_avg_dict[loss_type]
      if isinstance(loss_val, torch.Tensor):
        scales[loss_type] = loss_val.item()
      else:
        scales[loss_type] = float(loss_val)
    else:
      scales[loss_type] = 1.0

  # 找到最大尺度
  non_zero_scales = [v for v in scales.values() if v > 1e-8]
  if len(non_zero_scales) == 0:
    return base_weights.copy()

  max_scale = max(non_zero_scales)

  # 调整权重
  adjusted_weights = {}
  for loss_type, weight in base_weights.items():
    if loss_type not in scales or weight == 0:
      adjusted_weights[loss_type] = weight
      continue

    if scales[loss_type] > 1e-8:
      # 计算调整因子
      adjustment_factor = max_scale / scales[loss_type]

      # ✅ 限制调整倍数
      adjustment_factor = min(adjustment_factor, max_adjustment)

      adjusted_weights[loss_type] = weight * adjustment_factor
    else:
      adjusted_weights[loss_type] = weight

  return adjusted_weights


def train_one_epoch(
        is_training, loss_types, writer, step_index,
        save_tracking_results,
        seq_eval, cav_id_list,
        det_root, eval_dir_dict, save_dir, result_sha,
        cfg, cat, ID_start, hw, log,
        dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict,
        optimizer, len_record, detection_feature_path, training_ground_truth_data_all, det_id2str,
        num_frames_backprop, num_frames_per_sub_seq, force_gt_as_predicted_track, use_static_default_R,
        use_multiple_nets, show_vis, save_vis, training_split,
        regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, clip_grad_norm,
        parameters_list, delay_config, q_learning_config, enable_learnable_Q=None, q_net_dict=None,
        trajectory_q_loss_weight=None,  # 新增参数
        rq_consistency_loss_weight=None
):
  '''
  Train one epoch
  When is_training is False, then optimizer is None,
  this function is called by evaluation, num_frames_backprop = num_frames_per_sub_seq = -1
  and we do not call loss.backprop() or optimizer.step(),
  only finish tracking and save the results

  Output:
    loss_stats_avg: average loss over all seqs of one epoch
    loss_stats_avg_dict: details of different types of loss
  '''
  if not is_training:
    assert (num_frames_backprop == -1)
    assert (num_frames_per_sub_seq == -1)

  len_per_seq_eval = get_len_per_seq_eval(len_record, seq_eval)
  # During training, each sub seq has length == num_frames_backprop
  # During inference, each sub seq is a seq
  seq_name_list, seq_idx_in_eval_list, start_frame_idx_list, end_frame_idx_list = get_sub_seq_list_info(is_training,
                                                                                                        num_frames_per_sub_seq,
                                                                                                        len_record,
                                                                                                        len_per_seq_eval,
                                                                                                        seq_eval)
  num_sub_seqs = len(seq_name_list)
  print('num_sub_seqs: ', num_sub_seqs)

  epoch_q_stats = {
    'sequences': []
  }

  # loss for stats over one epoch's full sequence
  loss_stats_list_dict = {
    loss_type: {
      'sum': [],
      'count': 0
    } for loss_type in loss_types
  }

  # ========== 添加这里 ==========
  # 用于收集每个序列的跟踪数据
  sequence_frame_data = []
  current_seq_name = None
  # ================================

  # loop every sub sequence
  sub_seq_count = 0
  total_time, total_frames = 0.0, 0

  # random shuffle if training
  if is_training:
    sample_sub_seq_indices = np.random.permutation(num_sub_seqs)
  else:
    # no random shuffle
    sample_sub_seq_indices = range(num_sub_seqs)

  # for seq_idx_in_eval in range(len(seq_eval)):
  # for sub_seq_idx in range(num_sub_seqs):

  for sub_seq_idx in sample_sub_seq_indices:
    # print('sub_seq_idx: ', sub_seq_idx)

    # seq_name = seq_eval[seq_idx_in_eval]
    seq_name = seq_name_list[sub_seq_idx]
    seq_idx_in_eval = seq_idx_in_eval_list[sub_seq_idx]
    # ========== 添加这里：检测序列切换 ==========
    if current_seq_name is not None and current_seq_name != seq_name:
      # 保存上一个序列的数据
      # if save_vis and len(sequence_frame_data) > 0:
      if save_vis and len(sequence_frame_data) > 0:
        save_vis_dir = os.path.join(save_dir, 'visualization')
        save_frame_tracking_results(save_vis_dir, current_seq_name, sequence_frame_data)
        sequence_frame_data = []  # 清空

    current_seq_name = seq_name
    # ============================================
    # if seq_name == '0000' or seq_name == '0001' or seq_name == '0002' or seq_name == '0003' or seq_name == '0004' or seq_name == '0005' or seq_name == '0006':  # 跳过第一个场景
    #     continue
    seq_dets_dict = {}
    for cav_id in cav_id_list:
      seq_file = os.path.join(det_root, cav_id, seq_name + '.txt')
      # print('seq_file: ', seq_file)
      seq_dets, flag = load_detection(seq_file)  # load detection
      # print('flag: ', flag)
      if flag:
        seq_dets_dict[cav_id] = seq_dets
    if len(seq_dets_dict) == 0:
      continue  # no detection
    # print('seq_dets_dict: ', seq_dets_dict)

    # create folders for saving
    eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = \
      get_saving_dir(eval_dir_dict, seq_name, save_dir, cfg.num_hypo)

    # initialize tracker for training
    tracker, frame_list = initialize_DMSTrack(cfg, seq_name, cat, ID_start, hw, log,
                                              dtype, device,
                                              differentiable_kalman_filter_config,
                                              observation_covariance_net_dict,
                                              force_gt_as_predicted_track,
                                              use_static_default_R,
                                              use_multiple_nets,
                                              training_split,
                                              delay_config,
                                              q_learning_config,
                                              enable_learnable_Q=enable_learnable_Q,
                                              q_net_dict=q_net_dict)
    # print('frame_list: ', frame_list)

    # reset optimizer and loss
    if is_training:
      optimizer.zero_grad()
    # actual loss for backprop
    loss_list_dict = {
      loss_type: {
        'sum': [],
        'count': 0
      } for loss_type in loss_types
    }

    # loop over frame
    # min_frame, max_frame = int(frame_list[0]), int(frame_list[-1])
    min_frame = start_frame_idx_list[sub_seq_idx]
    max_frame = end_frame_idx_list[sub_seq_idx]
    # print('min_frame: ', min_frame)
    # print('max_frame: ', max_frame)

    # MY_DEBUG
    # max_frame = min_frame
    start_time = time.time()
    # tracer.start()
    for frame in range(min_frame, max_frame + 1):
      if not is_training:
        print_str = 'processing %s %s: %d/%d, %d/%d   \r' % (result_sha, seq_name, sub_seq_count, \
                                                             num_sub_seqs, frame, max_frame)
        sys.stdout.write(print_str)
        sys.stdout.flush()

      # for accessing single frame object feature of v2v4real
      global_timestamp_index = get_global_timestamp_index(seq_name, frame, len_record)

      # tracking by detection
      dets_frame_dict = {}
      dets_feature_dict = {}
      transformation_matrix_dict = {}
      for cav_id in seq_dets_dict.keys():
        dets_frame_dict[cav_id] = get_frame_det(seq_dets_dict[cav_id],
                                                frame)  # 处理指定列数据（1-6列；7-13列：包括物体的3D中心坐标 (x, y, z)、尺寸 (宽, 高, 长) 和旋转角度）
        if 'multi_sensor_differentiable_kalman_filter' in cfg.det_name:
          dets_feature_dict[cav_id] = load_detection_feature(global_timestamp_index, detection_feature_path, cav_id)
        else:
          dets_feature_dict[cav_id] = load_detection_feature_cobevt_single_sensor(global_timestamp_index,
                                                                                  detection_feature_path, cav_id)
        transformation_matrix_dict[cav_id] = load_transformation_matrix(global_timestamp_index, detection_feature_path,
                                                                        cav_id)

      # get ground-truth tensor data for this frame during training
      gt_boxes, gt_ids = get_ground_truth_data_single_frame(seq_idx_in_eval, frame, training_ground_truth_data_all,
                                                            dtype, device)

      since = time.time()
      # print('1 cav_id_list: ', cav_id_list)
      results, affi, loss_dict, matched_detection_id_dict, learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict = tracker.track_multi_sensor_differentiable_kalman_filter(
        dets_frame_dict, frame, seq_name, cav_id_list, dets_feature_dict, gt_boxes, gt_ids, transformation_matrix_dict)
      # results, affi, loss_dict, matched_detection_id_dict, learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict = tracker.track_multi_agent_with_mix_ci(
      #   dets_frame_dict, frame, seq_name, cav_id_list, dets_feature_dict, gt_boxes, gt_ids, transformation_matrix_dict)
      total_time += time.time() - since
      # if not is_training:  # 只在评估时打印
      #   print(f"\n[Sequence {seq_name} Inference Speed]")
      #   tracker.print_inference_speed_report()
      # print(total_time)
      # print('loss_dict: ', loss_dict)
      for loss_type in loss_types:
        # actual loss
        loss_list_dict[loss_type]['sum'].append(loss_dict[loss_type]['sum'])
        loss_list_dict[loss_type]['count'] += loss_dict[loss_type]['count']
        # loss stats
        loss_stats_list_dict[loss_type]['sum'].append(loss_dict[loss_type]['sum'].detach().cpu().numpy())
        loss_stats_list_dict[loss_type]['count'] += loss_dict[loss_type]['count']

      # print('loss_stats_list_dict: ', loss_stats_list_dict)

      # loss backprop and optimize
      loss_backprop_at_this_frame = False
      # comment out old code
      # if num_frames_backprop == -1 or frame == max_frame:
      #  if frame == max_frame:
      #    loss_backprop_at_this_frame = True
      # else:
      #  if frame != 0 and (frame % num_frames_backprop == 0 or frame == max_frame):
      #    loss_backprop_at_this_frame = True

      # loss backprop at end of each sub seq
      if frame == max_frame:
        loss_backprop_at_this_frame = True
      # or we reach the targeted number of frames
      if (frame - min_frame + 1) % num_frames_backprop == 0:
        loss_backprop_at_this_frame = True

      # if called by evaluate(), do not backprop
      if not is_training:
        loss_backprop_at_this_frame = False

      # MY_DEBUG
      # force no learning
      # loss_backprop_at_this_frame = False
      # print('loss_backprop_at_this_frame: ', loss_backprop_at_this_frame)
      # assert False

      # print('loss_backprop_at_this_frame: ', loss_backprop_at_this_frame)

      if loss_backprop_at_this_frame:
        loss_avg_dict = {}
        for loss_type in loss_types:
          # loss_all = torch.stack(loss_list_dict[loss_type]['sum'])
          # #print('loss_all: ', loss_all)
          # loss_sum = torch.sum(loss_all)
          loss_sum = torch.tensor(0.0, dtype=dtype, device=device)
          for tensor in loss_list_dict[loss_type]['sum']:
            loss_sum += torch.sum(tensor)  # torch.sum()能处理任意形状的张量
          # print('loss_sum: ', loss_sum)
          if loss_list_dict[loss_type]['count'] == 0:
            loss_list_dict[loss_type]['count'] += 1
          loss_avg = loss_sum / loss_list_dict[loss_type]['count']
          # print('loss_avg: ', loss_avg)
          loss_avg_dict[loss_type] = loss_avg

        regression_loss_avg = loss_avg_dict['regression']
        association_loss_avg = loss_avg_dict['association']
        det_neg_log_likelihood_loss_avg = loss_avg_dict['det_neg_log_likelihood']
        # ====================添加trajectory_q_loss损失和trajectory_s_loss损失==================================
        # ✅ 先定义 base_weights（在 if-else 之前）
        base_weights = {
          'regression': regression_loss_weight,
          'association': association_loss_weight,
          'det_neg_log_likelihood': det_neg_log_likelihood_loss_weight,
          'trajectory_q_loss': trajectory_q_loss_weight if enable_learnable_Q else 0,
          'trajectory_s_loss': rq_consistency_loss_weight if enable_learnable_Q else 0
        }
        if enable_learnable_Q and ('trajectory_q_loss' in loss_avg_dict or 'trajectory_s_loss' in loss_avg_dict):
          trajectory_q_loss_avg = loss_avg_dict.get('trajectory_q_loss',
                                                    torch.tensor(0.0, dtype=dtype, device=device))
          trajectory_s_loss = loss_avg_dict.get('trajectory_s_loss', torch.tensor(0.0, dtype=dtype, device=device))

          # 总Q损失权重（可以调整）

          # 计算自适应权重
          adjusted_weights = compute_adaptive_loss_weights(loss_avg_dict, base_weights, dtype, device)

          # 使用调整后的权重计算总损失
          loss_avg = (loss_avg_dict['regression'] * adjusted_weights['regression'] +
                      loss_avg_dict['association'] * adjusted_weights['association'] +
                      loss_avg_dict['det_neg_log_likelihood'] * adjusted_weights['det_neg_log_likelihood'] +
                      loss_avg_dict['trajectory_q_loss'] * adjusted_weights['trajectory_q_loss'] +
                      loss_avg_dict['trajectory_s_loss'] * adjusted_weights['trajectory_s_loss'])

          print(f'\n{"=" * 80}')
          print(f'Frame {frame} - Adaptive Loss Weights:')
          print(f'  Original weights: {base_weights}')
          print(f'  Adjusted weights: {adjusted_weights}')
          print(f'  Loss values: regression={regression_loss_avg.item():.6f}, '
                f'association={association_loss_avg.item():.6f}, '
                f'det_neg_log={det_neg_log_likelihood_loss_avg.item():.6f}, '
                f'traj_q={trajectory_q_loss_avg.item():.6f}, '
                f'traj_s={trajectory_s_loss.item():.6f}')
          print(f'  Weighted contributions: '
                f'reg={loss_avg_dict["regression"].item() * adjusted_weights["regression"]:.6f}, '
                f'assoc={loss_avg_dict["association"].item() * adjusted_weights["association"]:.6f}, '
                f'det={loss_avg_dict["det_neg_log_likelihood"].item() * adjusted_weights["det_neg_log_likelihood"]:.6f}, '
                f'q={loss_avg_dict["trajectory_q_loss"].item() * adjusted_weights["trajectory_q_loss"]:.6f}, '
                f's={loss_avg_dict["trajectory_s_loss"].item() * adjusted_weights["trajectory_s_loss"]:.6f}')
          print(f'  Total loss: {loss_avg.item():.6f}')
          print(f'{"=" * 80}\n')

        else:
          # 不使用Q损失时的原始计算
          loss_avg = (regression_loss_avg * regression_loss_weight +
                      association_loss_avg * association_loss_weight +
                      det_neg_log_likelihood_loss_avg * det_neg_log_likelihood_loss_weight)

          print('frame: %d, actual_loss: %f, det_neg_log_likelihood_loss: %f, '
                'regression_loss: %f, association_loss: %f' %
                (frame, loss_avg.item(), det_neg_log_likelihood_loss_avg.item(),
                 regression_loss_avg.item(), association_loss_avg.item()))

        tensorboard_loss_dict = {loss_type: loss_avg_dict[loss_type].item() for loss_type in loss_types}
        write_tensorboard(writer, 'step', step_index, loss_types,
                          None, loss_avg.item(), tensorboard_loss_dict,
                          None, None, None)

        if not use_static_default_R:
          loss_avg.backward()
          if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters_list, clip_grad_norm)
          optimizer.step()
          step_index += 1
        # reset
        # reset optimizer and loss
        optimizer.zero_grad()
        loss_list_dict = {
          loss_type: {
            'sum': [],
            'count': 0
          } for loss_type in loss_types
        }

        # reset dkf model gradients
        tracker.reset_dkf_gradients()

      if save_tracking_results:
        # saving affinity matrix, between the past frame and current frame
        # e.g., for 000006.npy, it means affinity between frame 5 and 6
        # note that the saved value in affinity can be different in reality because it is between the
        # original detections and ego-motion compensated predicted tracklets, rather than between the
        # actual two sets of output tracklets
        save_affi_file = os.path.join(affinity_dir, '%06d.npy' % frame)
        save_affi_vis = os.path.join(affinity_vis, '%06d.txt' % frame)
        if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0):
          # save affinity as long as there are tracklets in at least one frame
          np.save(save_affi_file, affi)
          # cannot save for visualization unless both two frames have tracklets
          if affi.shape[0] > 0 and affi.shape[1] > 0:
            save_affinity(affi, save_affi_vis)
        # saving trajectories, loop over each hypothesis
        for hypo in range(cfg.num_hypo):
          save_trk_file = os.path.join(save_trk_dir[hypo], '%06d.txt' % frame)
          save_trk_file = open(save_trk_file, 'w')
          for result_tmp in results[hypo]:  # N x 15
            save_results(result_tmp, save_trk_file, eval_file_dict[hypo], \
                         det_id2str, frame, cfg.score_threshold)
          save_trk_file.close()

      if show_vis or save_vis:
        if save_vis:
          sequence_frame_data.append({
            'frame': frame,
            'tracking_results': results[0].copy(),  # 假设使用第一个hypothesis
            'gt_boxes': gt_boxes.detach().cpu().numpy(),
            'gt_ids': gt_ids.detach().cpu().numpy(),
            'track_P': track_P
          })
        if seq_name == '0008':
          if save_vis and len(sequence_frame_data) > 0:
            save_vis_dir = os.path.join(save_dir, 'visualization')
            save_frame_tracking_results(save_vis_dir, current_seq_name, sequence_frame_data)
          assert (len(results) == 1)
        # MY_DEBUG
        if True:
          # 删除特定帧的特定ID
          ids_to_delete = {
            101: [26],
            102: [26, 43]  # 第101帧删除ID为3和5的跟踪框

          }
          manual_adjustments = {
            # 32: {  # 第101帧
            #   # 1: {'dx': 0.0, 'dy': -1.0, 'dz': 0.0, 'dtheta': 0.1},
            #   0: {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1},
            #   # 10: {'dx': 2.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1}
            # },
            # 33: {  # 第102帧
            #   0: {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1},
            #   # 10: {'dx': 2.0, 'dy': 0.0, 'dz': 0.0, 'dtheta': 0.1}
            # }
          }
          # if frame >=15:
          save_vis_dir = os.path.join(save_dir, 'visualization')
          mkdir_if_missing(save_vis_dir)
          visualize(
            show_vis, save_vis, save_vis_dir,
            results[0],
            gt_boxes.detach().cpu().numpy(),
            gt_ids.detach().cpu().numpy(),
            dets_frame_dict,
            detection_feature_path, seq_name, frame, global_timestamp_index,
            matched_detection_id_dict, learnable_R_dict, track_P, manual_adjustments=manual_adjustments,
            ids_to_delete=ids_to_delete)

      # end of this frame
      sys.stdout.flush()
      total_frames += 1
    # tracer.stop()
    # tracer.save()
    end = time.time()
    print('frame: %d, total time: %f' % (total_frames, end - start_time))
    # tracer.stop()
    # tracer.save()
    #
    # end of this sub seq
    # tracer.stop()
    # tracer.save()
    # ✅ 添加这里：序列结束后收集统计
    seq_stats = tracker.get_final_q_statistics()
    if seq_stats:
      seq_stats['seq_name'] = seq_name
      epoch_q_stats['sequences'].append(seq_stats)
    sys.stdout.flush()
    sub_seq_count += 1

    for index in range(cfg.num_hypo):
      eval_file_dict[index].close()
      ID_start = max(ID_start, tracker.ID_count[index])

    # print_log('%s, %25s: %4.f seconds for %5d frames or %6.1f FPS, metric is %s = %.2f' % \
    #  (cfg.dataset, result_sha, total_time, total_frames, total_frames / total_time, \
    #  tracker.metric, tracker.thres), log=log)

  speed = total_frames / total_time
  print('Tracking speed: %f frames per second, total frames: %d, total time: %f' % (speed, total_frames, total_time))
  # assert False

  # print loss stats at the end of epoch
  # MY_DEBUG
  # if frame == max_frame:
  if True:
    loss_stats_avg_dict = {}
    for loss_type in loss_types:
      # 确保所有损失值都是标量
      loss_stats_sum_list = []
      for loss_val in loss_stats_list_dict[loss_type]['sum']:
        # 强制转换为标量
        if hasattr(loss_val, 'item'):
          scalar_val = loss_val.item()
        elif np.isscalar(loss_val):
          scalar_val = float(loss_val)
        else:
          scalar_val = float(np.sum(loss_val))  # 如果是数组，求和后转标量
        loss_stats_sum_list.append(scalar_val)
      loss_stats_sum = np.sum(loss_stats_sum_list)
      # loss_stats_all = np.stack(loss_stats_list_dict[loss_type]['sum'])
      # loss_stats_sum = np.sum(loss_stats_all)
      if loss_stats_list_dict[loss_type]['count'] == 0:
        loss_stats_list_dict[loss_type]['count'] += 1
      loss_stats_avg = loss_stats_sum / loss_stats_list_dict[loss_type]['count']
      loss_stats_avg_dict[loss_type] = loss_stats_avg

    regression_loss_stats_avg = loss_stats_avg_dict['regression']
    association_loss_stats_avg = loss_stats_avg_dict['association']
    det_neg_log_likelihood_loss_stats_avg = loss_stats_avg_dict['det_neg_log_likelihood']
    if enable_learnable_Q:
      trajectory_q_loss_stats_avg = loss_stats_avg_dict['trajectory_q_loss']
      rq_consistency_loss_stats_avg = loss_stats_avg_dict['trajectory_s_loss']
    # process_noise_loss_avg = loss_stats_avg_dict['process_noise']
    if enable_learnable_Q:
      loss_stats_avg = regression_loss_stats_avg * regression_loss_weight + association_loss_stats_avg * association_loss_weight + det_neg_log_likelihood_loss_stats_avg * det_neg_log_likelihood_loss_weight + trajectory_q_loss_stats_avg * trajectory_q_loss_weight + rq_consistency_loss_stats_avg * rq_consistency_loss_weight
      print(
        'frame: %d,  loss_stats: %f, det_neg_log_likelihood_loss: %f, regression_loss: %f, association_loss: %f，trajectory_q_loss: %f, trajectory_s_loss: %f' % (
        frame, loss_stats_avg.item(), det_neg_log_likelihood_loss_stats_avg.item(), regression_loss_stats_avg.item(),
        association_loss_stats_avg.item(), trajectory_q_loss_stats_avg.item(), rq_consistency_loss_stats_avg.item()))

    else:
      loss_stats_avg = regression_loss_stats_avg * regression_loss_weight + association_loss_stats_avg * association_loss_weight + det_neg_log_likelihood_loss_stats_avg * det_neg_log_likelihood_loss_weight
      print('frame: %d,  loss_stats: %f, det_neg_log_likelihood_loss: %f, regression_loss: %f, association_loss: %f' % (
      frame, loss_stats_avg.item(), det_neg_log_likelihood_loss_stats_avg.item(), regression_loss_stats_avg.item(),
      association_loss_stats_avg.item()))

    # reset
    # loss_stats_list_dict = {
    #  loss_type: {
    #    'sum': [],
    #    'count': 0
    #  } for loss_type in loss_types
    # }
  # end of this epoch
  sys.stdout.flush()

  # ✅ 添加这里：打印epoch统计
  if len(epoch_q_stats['sequences']) > 0:
    print("\n" + "=" * 80)
    mode_str = "TRAINING" if is_training else "EVALUATION"
    print(f"EPOCH Q USAGE SUMMARY ({mode_str})")
    print("=" * 80)

    total_preds = sum(s['total_predictions'] for s in epoch_q_stats['sequences'])
    total_adaptive = sum(s['adaptive_q_used'] for s in epoch_q_stats['sequences'])
    total_default = sum(s['default_q_used'] for s in epoch_q_stats['sequences'])

    if total_preds > 0:
      print(f"Total predictions across all sequences: {total_preds}")
      print(f"Adaptive Q used: {total_adaptive} ({total_adaptive / total_preds * 100:.2f}%)")
      print(f"Default Q used: {total_default} ({total_default / total_preds * 100:.2f}%)")
      print(f"\nPer-sequence breakdown:")

      for seq_stat in epoch_q_stats['sequences']:
        print(f"  {seq_stat['seq_name']}: "
              f"{seq_stat['adaptive_q_percentage']:.1f}% adaptive "
              f"({seq_stat['adaptive_q_used']}/{seq_stat['total_predictions']})")
    else:
      print("No predictions made (all sequences empty?)")
    print("=" * 80)

  return loss_stats_avg, loss_stats_avg_dict, step_index


def get_evaluation_metrics(evaluation_file_save_dir):
  print('evaluation_file_save_dir: ', evaluation_file_save_dir)
  # ./results/v2v4real/debug_evaluation_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_0
  evaluation_summary_file = os.path.join(evaluation_file_save_dir, 'summary_car_average_eval3D.txt')
  metrics_dict = {}

  # get the line right below ' sAMOTA  AMOTA  AMOTP'
  found_metrics_line = False
  with open(evaluation_summary_file, 'r') as f:
    for line in f:
      line = line.strip()
      print(line)
      if found_metrics_line:
        metrics_dict['sAMOTA'] = float(line[0])
        metrics_dict['AMOTA'] = float(line[1])
        metrics_dict['AMOTP'] = float(line[2])
        break

      if len(line) > 0 and line[0] == 'sAMOTA':
        found_metrics_line = True

  print('metrics_dict: ', metrics_dict)
  assert False
  return metrics_dict


# Using the model to run tracking and evaluation on the evaluation split, without loss backprop
def track_and_evaluate(save_dir_prefix, cav_id_list, observation_covariance_net_dict, evaluation_ground_truth_data_all,
                       evaluation_config_dict, evaluation_split, cfg, seq_eval_mode, epoch_idx, cat, ID_start, hw, log,
                       dtype, device, differentiable_kalman_filter_config, det_id2str, force_gt_as_predicted_track,
                       use_static_default_R, use_multiple_nets, show_vis, save_vis, mail, threshold_3D_IOU, loss_types,
                       writer, regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight,
                       q_learning_config, enable_learnable_Q=False, q_net_dict=None, trajectory_q_loss_weight=None,
                       rq_consistency_loss_weight=None, is_training=True):
  '''
  Similar to the code calling train_one_epoch, but do not call loss.backprop() and optimizer.step()
  '''
  with torch.no_grad():
    for cav_id, model in observation_covariance_net_dict.items():
      model.eval()
    if enable_learnable_Q and not is_training:
      for cav_id, q_net in q_net_dict.items():
        q_net.eval()
        print(f"Set Q-net {cav_id} to eval mode: {not q_net.training}")
    #             # ✅ 验证 eval 模式
    #             print(f"\n=== Q-Net {cav_id} Status ===")
    #             print(f"Training mode: {q_net.training}")  # 应该是 False

    #             # 检查 dropout 状态
    #             for name, module in q_net.named_modules():
    #                 if isinstance(module, nn.Dropout):
    #                     print(f"  Dropout {name}: training={module.training}")
    #                 if isinstance(module, nn.LSTM):
    #                     print(f"  LSTM {name}: training={module.training}")
    print('epoch_idx: ', epoch_idx)

    evaluation_save_folder = 'evaluation_' + evaluation_config_dict[
      'result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % epoch_idx
    evaluation_save_folder = os.path.join(save_dir_prefix, evaluation_save_folder)
    evaluation_save_dir = os.path.join(cfg.save_root, evaluation_save_folder);
    mkdir_if_missing(evaluation_save_dir)
    print('evaluation_save_dir: ', evaluation_save_dir)
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_all_H1_epoch_0
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_0001_H1_epoch_0

    # create eval dir for each hypothesis
    eval_dir_dict = dict()
    for index in range(cfg.num_hypo):
      eval_dir_dict[index] = os.path.join(evaluation_save_dir, 'data_%d' % index);
      mkdir_if_missing(eval_dir_dict[index])
    # print('eval_dir_dict[0]: ', eval_dir_dict[0])
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0/data_0

    save_tracking_results = True
    # set is_training = False, when calling train_one_epoch
    # optimizer = None
    # also set num_frames_backprop = -1
    # num_frames_per_sub_seq = -1
    # set step_index = -1 because optimizer.step() will not be called
    # the returned step_index will not be used
    # set clip_grad_norm = -1, we will not call optimizer.step()
    # set parameters_list = []
    evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict, _ = train_one_epoch(
      False, loss_types, writer, -1,
      save_tracking_results,
      evaluation_config_dict['seq_eval'], cav_id_list,
      evaluation_config_dict['det_root'], eval_dir_dict, evaluation_save_dir, evaluation_config_dict['result_sha'],
      cfg, cat, ID_start, hw, log,
      dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict,
      None, evaluation_config_dict['len_record'], evaluation_config_dict['detection_feature_path'],
      evaluation_ground_truth_data_all, det_id2str,
      -1, -1, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets, show_vis, save_vis,
      evaluation_split,
      regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, -1, [], delay_config, q_learning_config,
      enable_learnable_Q, q_net_dict,
      trajectory_q_loss_weight,  # 新增参数
      rq_consistency_loss_weight)

    # =================保存过程噪声网络检查点===============================
    if enable_learnable_Q:

      for cav_id, model in q_net_dict.items():
        # model.eval()
        trajectory_q_net_model_weight = os.path.join(evaluation_save_dir,
                                                     f'traj_based_q_net_model_{cav_id}_epoch_{epoch_idx}.pth')
        torch.save(q_net_dict[cav_id].state_dict(), trajectory_q_net_model_weight)

    # save model checkpoint
    for cav_id in cav_id_list:
      model_save_file = os.path.join(evaluation_save_dir, 'model_%s_epoch_%d.pth' % (cav_id, epoch_idx))
      torch.save(observation_covariance_net_dict[cav_id].state_dict(), model_save_file)

    run_evaluation_every_epoch = True
    if run_evaluation_every_epoch:
      # 'all', '0000', '0001', ...
      # this command will evaluate the current evaluation split's tracking results
      # evaluation_split_evaluation_command = 'python3 scripts/KITTI/evaluate.py ' + evaluation_save_folder + ' 1 3D 0.25 %s %s' % (seq_eval_mode, evaluation_split)
      # print('evaluation_split_evaluation_command: ', evaluation_split_evaluation_command)
      # returned_value = os.system(evaluation_split_evaluation_command)
      # print('returned_value: ', returned_value)

      # directly call ab3dmot evaluate()
      evaluation_metrics_dict = evaluate(evaluation_save_folder, mail, 1, True, False, threshold_3D_IOU, True,
                                         seq_eval_mode, evaluation_split)

    for cav_id, model in observation_covariance_net_dict.items():
      model.train()

  sys.stdout.flush()
  return evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict


def get_config(cfg, cat, training_split, evaluation_split, seq_eval_mode, use_reversed_cav_id_list):
  '''
  Input:
    cfg.det_name
    cfg.data_set
    training_split: can be 'train' or 'val'
    evaluation_split: can be 'train' or 'val'
    seq_eval_mode: one of ['all', '0001', '0002', ...]

  Output:

    training_config_dict = {
      result_sha :
      det_root :
      seq_eval :
      len_record :
      detection_feature_path :
    }

    evaluation_config_dict has the same structure as above

    det_id2str
    hw
    data_root
    cav_id_list
    differentiable_kalman_filter_config
    learning_rate

  '''
  # common config
  det_id2str = {2: 'Car'}
  hw = {'image': None, 'lidar': None}
  data_root = '../AB3DMOT/data/v2v4real'
  if use_reversed_cav_id_list:
    cav_id_list = ['1', 'ego']
  else:
    cav_id_list = ['ego', '1']

  if 'multi_sensor_differentiable_kalman_filter' not in cfg.det_name:
    # single sensor, such as cobevt + dkf
    cav_id_list = ['ego']

  if 'multi_sensor_differentiable_kalman_filter' in cfg.det_name:
    dkf_type = 'multi_sensor'
    feature_channel_size = 320
    feature_region_size = 20
  else:  # cobevt + dkf
    dkf_type = 'single_sensor'
    feature_channel_size = 512
    feature_region_size = 5
  differentiable_kalman_filter_config = {
    'dim_x': 10,
    'dim_z': 7,
    'dkf_type': dkf_type,
    'observation_covariance_setting': {
      'feature_channel_size': feature_channel_size,
      'feature_region_size': feature_region_size
    },
    'gt_data_association_threshold': 4  # 4 meters center distance
  }
  learning_rate = 1e-3
  weight_decay = 1e-5

  train_result_sha = '%s_%s_%s' % (cfg.det_name, cat, 'train')
  train_det_root = os.path.join('../AB3DMOT/data', cfg.dataset, 'detection', train_result_sha)

  val_result_sha = '%s_%s_%s' % (cfg.det_name, cat, 'val')
  val_det_root = os.path.join('../AB3DMOT/data', cfg.dataset, 'detection', val_result_sha)

  train_seq_eval = [
    '0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',
    '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
    '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029',
    '0030', '0031'
  ]
  train_len_record = [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 4255,
                      4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]

  val_seq_eval = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
  val_len_record = [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]

  if 'multi_sensor_differentiable_kalman_filter' in cfg.det_name:
    train_detection_feature_path = '../V2V4Real/official_models/train_no_fusion_keep_all/npy/'
    val_detection_feature_path = '../V2V4Real/official_models/no_fusion_keep_all/npy/'
  else:  # cobevt + dkf
    train_detection_feature_path = '../V2V4Real/official_models/train_cobevt/npy/'
    val_detection_feature_path = '../V2V4Real/official_models/cobevt/npy/'

  train_config_dict = {
    'result_sha': train_result_sha,
    'det_root': train_det_root,
    'seq_eval': train_seq_eval if seq_eval_mode == 'all' else [seq_eval_mode],
    'len_record': train_len_record,
    'detection_feature_path': train_detection_feature_path
  }
  val_config_dict = {
    'result_sha': val_result_sha,
    'det_root': val_det_root,
    'seq_eval': val_seq_eval if seq_eval_mode == 'all' else [seq_eval_mode],
    'len_record': val_len_record,
    'detection_feature_path': val_detection_feature_path,
  }

  training_config_dict = train_config_dict if training_split == 'train' else val_config_dict
  evaluation_config_dict = val_config_dict if evaluation_split == 'val' else train_config_dict

  return training_config_dict, evaluation_config_dict, det_id2str, hw, data_root, cav_id_list, differentiable_kalman_filter_config, learning_rate, weight_decay


def write_tensorboard(writer, freq, freq_idx, loss_types,
                      training_metrics_dict, training_loss_stats_avg, training_loss_stats_avg_dict,
                      evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict):
  '''
  Input
    freq: one of ['epoch', 'step']
    freq_idx: epoch index or step index
  '''

  if training_metrics_dict is not None:
    writer.add_scalar('training/%s/sAMOTA' % freq, training_metrics_dict['sAMOTA'], freq_idx)
    writer.add_scalar('training/%s/AMOTA' % freq, training_metrics_dict['AMOTA'], freq_idx)
    writer.add_scalar('training/%s/AMOTP' % freq, training_metrics_dict['AMOTP'], freq_idx)

  if training_loss_stats_avg is not None:
    writer.add_scalar('training/%s/loss' % freq, training_loss_stats_avg, freq_idx)
    for loss_type in loss_types:
      writer.add_scalar('training/%s/%s_loss' % (freq, loss_type), training_loss_stats_avg_dict[loss_type], freq_idx)

  if evaluation_metrics_dict is not None:
    writer.add_scalar('evaluation/%s/sAMOTA' % freq, evaluation_metrics_dict['sAMOTA'], freq_idx)
    writer.add_scalar('evaluation/%s/AMOTA' % freq, evaluation_metrics_dict['AMOTA'], freq_idx)
    writer.add_scalar('evaluation/%s/AMOTP' % freq, evaluation_metrics_dict['AMOTP'], freq_idx)

  if evaluation_loss_stats_avg is not None:
    writer.add_scalar('evaluation/%s/loss' % freq, evaluation_loss_stats_avg, freq_idx)
    for loss_type in loss_types:
      writer.add_scalar('evaluation/%s/%s_loss' % (freq, loss_type), evaluation_loss_stats_avg_dict[loss_type],
                        freq_idx)

  # # =====================添加过程噪声损失记录===============================
  # if 'process_noise' in loss_types:
  #   if training_loss_stats_avg_dict and 'process_noise' in training_loss_stats_avg_dict:
  #     writer.add_scalar('training/%s/process_noise_loss' % freq, training_loss_stats_avg_dict['process_noise'],
  #                       freq_idx)
  #
  #   if evaluation_loss_stats_avg_dict and 'process_noise' in evaluation_loss_stats_avg_dict:
  #     writer.add_scalar('evaluation/%s/process_noise_loss' % freq, evaluation_loss_stats_avg_dict['process_noise'],
  #                       freq_idx)
  # =====================添加trajectory_q_loss和rq_consistency损失记录===============================
  if 'trajectory_q_loss' in loss_types:
    if training_loss_stats_avg_dict and 'trajectory_q_loss' in training_loss_stats_avg_dict:
      writer.add_scalar('training/%s/trajectory_q_loss' % freq,
                        training_loss_stats_avg_dict['trajectory_q_loss'], freq_idx)

    if evaluation_loss_stats_avg_dict and 'trajectory_q_loss' in evaluation_loss_stats_avg_dict:
      writer.add_scalar('evaluation/%s/trajectory_q_loss' % freq,
                        evaluation_loss_stats_avg_dict['trajectory_q_loss'], freq_idx)

  if 'rq_consistency' in loss_types:
    if training_loss_stats_avg_dict and 'rq_consistency' in training_loss_stats_avg_dict:
      writer.add_scalar('training/%s/rq_consistency_loss' % freq,
                        training_loss_stats_avg_dict['rq_consistency'], freq_idx)

    if evaluation_loss_stats_avg_dict and 'rq_consistency' in evaluation_loss_stats_avg_dict:
      writer.add_scalar('evaluation/%s/rq_consistency_loss' % freq,
                        evaluation_loss_stats_avg_dict['rq_consistency'], freq_idx)

  return


def _build_q_config(differentiable_kalman_filter_config, q_learning_method):
  """
  从differentiable_kalman_filter_config构造q_learning_config

  Args:
      differentiable_kalman_filter_config: 原始卡尔曼滤波配置
      q_learning_method: Q学习方法 ('innovation_based', 'physics_constrained', 'multiscale_attention')

  Returns:
      q_learning_config: Q学习配置字典
  """
  if differentiable_kalman_filter_config is not None:
    # 从原配置中提取基础信息
    dim_x = differentiable_kalman_filter_config.get('dim_x', 10)
    dim_z = differentiable_kalman_filter_config.get('dim_z', 7)

    # 获取观测协方差设置
    obs_cov_setting = differentiable_kalman_filter_config.get('observation_covariance_setting', {})
    feature_channel_size = obs_cov_setting.get('feature_channel_size', 512)
    feature_region_size = obs_cov_setting.get('feature_region_size', 5)

    # 构造Q学习配置
    q_learning_config = {
      # 基础维度信息
      'dim_x': dim_x,
      'dim_z': dim_z,

      # Q学习方法
      'q_learning_method': q_learning_method,

      # 网络配置
      'feature_channel_size': feature_channel_size,
      'feature_region_size': feature_region_size,

      # 学习参数
      'learning_rate': 1e-4,
      'history_length': 10,  # 历史序列长度
      'hidden_dim': 128,  # 隐藏层维度

      # 损失权重
      'loss_weights': {
        'innovation_consistency': 1.0,  # 创新一致性损失权重
        'regularization': 0.1,  # 正则化损失权重
        'positive_definiteness': 0.5,  # 正定性损失权重
        'physics_constraint': 0.3,  # 物理约束损失权重（如果使用）
        'attention_regularization': 0.2  # 注意力正则化权重（如果使用）
      },

      # 其他配置
      'gt_data_association_threshold': differentiable_kalman_filter_config.get(
        'gt_data_association_threshold', 3.0),
      'dkf_type': differentiable_kalman_filter_config.get('dkf_type', 'multi_sensor'),

      # 根据不同方法的特定配置
      # **self._get_method_specific_config(q_learning_method, differentiable_kalman_filter_config)
    }
  else:
    # 如果没有提供配置，使用默认值
    q_learning_config = {
      'dim_x': 10,
      'dim_z': 7,
      'q_learning_method': q_learning_method,
      'feature_channel_size': 512,
      'feature_region_size': 5,
      'learning_rate': 1e-4,
      'history_length': 10,
      'hidden_dim': 128,
      'loss_weights': {
        'innovation_consistency': 1.0,
        'regularization': 0.1,
        'positive_definiteness': 0.5,
        'physics_constraint': 0.3,
        'attention_regularization': 0.2
      },
      'gt_data_association_threshold': 3.0,
      'dkf_type': 'multi_sensor',
      # **get_method_specific_config(q_learning_method, None)
    }

  return q_learning_config


def main_per_cat_multi_sensor_differentiable_kalman_filter(cfg, cat, log, ID_start, dtype, device,
                                                           run_evaluation_every_epoch, seq_eval_mode,
                                                           force_gt_as_predicted_track, use_static_default_R,
                                                           use_multiple_nets, show_vis, save_vis, num_frames_backprop,
                                                           num_frames_per_sub_seq, num_epochs, load_model_path,
                                                           save_dir_prefix, training_split, evaluation_split, feature,
                                                           regression_loss_weight, association_loss_weight,
                                                           det_neg_log_likelihood_loss_weight, clip_grad_norm,
                                                           use_reversed_cav_id_list,
                                                           delay_config,
                                                           enable_learnable_Q=False,
                                                           q_learning_method='innovation_based',
                                                           trajectory_q_loss_weight=1,  # 新增参数
                                                           rq_consistency_loss_weight=1,
                                                           is_training=False,
                                                           ):
  # get data-cat-split specific path
  training_config_dict, evaluation_config_dict, det_id2str, hw, data_root, cav_id_list, differentiable_kalman_filter_config, learning_rate, weight_decay = get_config(
    cfg, cat, training_split, evaluation_split, seq_eval_mode, use_reversed_cav_id_list)
  print('training_config_dict: ', training_config_dict)
  print('evaluation_config_dict: ', evaluation_config_dict)

  # load ground truth label
  training_ground_truth_data_all = load_ground_truth_data_all(training_config_dict['result_sha'], seq_eval_mode,
                                                              training_split)
  evaluation_ground_truth_data_all = load_ground_truth_data_all(evaluation_config_dict['result_sha'], seq_eval_mode,
                                                                evaluation_split)

  # tensorboad setup
  # write tensorboard file in the training folder epoch_idx=0
  tensorboard_save_folder = 'training_' + training_config_dict[
    'result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % 0
  tensorboard_save_folder = os.path.join(save_dir_prefix, tensorboard_save_folder)
  tensorboard_save_dir = os.path.join(cfg.save_root, tensorboard_save_folder, 'tensorboard');
  mkdir_if_missing(tensorboard_save_dir)
  print('tensorboard_save_dir: ', tensorboard_save_dir)
  writer = SummaryWriter(tensorboard_save_dir)

  # model
  torch.autograd.set_detect_anomaly(True)
  observation_covariance_net_dict = {}
  traj_based_q_net_dict = {}
  # innovation_based_q_net = {}  # 新增
  for cav_id in cav_id_list:
    observation_covariance_net_dict[cav_id] = ObservationCovarianceNet(differentiable_kalman_filter_config, feature).to(
      device)
    q_learning_config = _build_q_config(differentiable_kalman_filter_config, q_learning_method)
    traj_based_q_net_dict[cav_id] = TrajectoryBasedQNet(q_learning_config, device, dtype).to(device)

    # ================过程噪声网络 (可以选择每个CAV单独一个或共享一个)======================
    # if enable_learnable_Q:
    #   if use_multiple_nets:
    #     # 每个CAV独立的过程噪声网络
    #     process_noise_net_dict[cav_id] = ProcessNoiseNet(
    #       differentiable_kalman_filter_config=differentiable_kalman_filter_config,
    #       device=device,
    #       dtype=dtype,
    #       history_length=10,
    #       hidden_dim=128
    #     ).to(device)
    #   else:
    #     # 共享过程噪声网络
    #     if 'shared' not in process_noise_net_dict:
    #       process_noise_net_dict['shared'] = ProcessNoiseNet(
    #         differentiable_kalman_filter_config=differentiable_kalman_filter_config,
    #         device=device,
    #         dtype=dtype,
    #         history_length=10,
    #         hidden_dim=128
    #       ).to(device)

    if load_model_path != '':
      model_file = load_model_path.replace('ego', cav_id)
      print('model_file: ', model_file)
      observation_covariance_net_dict[cav_id].load_state_dict(torch.load(model_file, map_location=device))
      # =======================冻结权重==============================

      # # 冻结BEV特征提取部分
      for param in observation_covariance_net_dict[cav_id].bev_conv_and_max_pool.parameters():
        param.requires_grad = False
      #
      # # 冻结位置编码部分
      # for param in observation_covariance_net_dict[cav_id].positional_encoding_linear.parameters():
      #   param.requires_grad = False
      #
      #   print(f"   已冻结 {cav_id} 网络的BEV和位置编码层")
      # ====================== 尝试加载过程噪声网络权重=====================
      # if enable_learnable_Q:
      #   traj_based_q_net_model_file = model_file.replace('model_', 'traj_based_q_net_model_')
      #   if os.path.exists(traj_based_q_net_model_file):
      #     print('Loading process noise model file: ', traj_based_q_net_model_file)
      #     traj_based_q_net_dict[cav_id].load_state_dict(torch.load(traj_based_q_net_model_file,map_location=device))
          # if use_multiple_nets:
          #   process_noise_net_dict[cav_id].load_state_dict(torch.load(process_noise_model_file))
          # else:
          #   process_noise_net_dict['shared'].load_state_dict(torch.load(process_noise_model_file))
      # MY_DEBUG

      if False:
        for param in observation_covariance_net_dict[cav_id].parameters():
          print(torch.mean(param.data))
          print(torch.min(param.data))  # 1e-6
          print(torch.max(param.data))  # 124
          print(torch.norm(param.data))
          assert False

  # optimizer
  parameters_list = list(observation_covariance_net_dict['ego'].parameters())
  if use_multiple_nets and '1' in cav_id_list:
    parameters_list = parameters_list + list(observation_covariance_net_dict['1'].parameters())

  # =============================添加traj_based_q_net网络参数===============================
  if enable_learnable_Q:
    parameters_list += list(traj_based_q_net_dict['ego'].parameters())
    if use_multiple_nets and '1' in cav_id_list:
      # parameters_list = parameters_list + list(innovation_based_q_net.parameters())
      parameters_list += list(traj_based_q_net_dict['1'].parameters())
    # if use_multiple_nets:
    #   for cav_id in cav_id_list:
    #     if cav_id in innovation_based_q_net:
    #       parameters_list = parameters_list + list(innovation_based_q_net.parameters())
    # else:
    #   if 'shared' in innovation_based_q_net:
    #     parameters_list = parameters_list + list(innovation_based_q_net.parameters())

  optimizer = torch.optim.Adam(parameters_list, lr=learning_rate, weight_decay=weight_decay)
  loss_types = ['regression', 'association', 'det_neg_log_likelihood']
  if enable_learnable_Q:
    # loss_types.append('process_noise')
    loss_types.extend(['trajectory_q_loss', 'trajectory_s_loss'])

  mail = mailpy.Mail('')
  threshold_3D_IOU = 0.25
  # initial evaluation before training loop
  # MY_DEBUG
  run_init_evaluation = True
  if run_init_evaluation:
    print('Run initial evaluation before training loop')
    # let epoch_idx = 0
    # start = time.time()
    evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict = track_and_evaluate(
      save_dir_prefix, cav_id_list, observation_covariance_net_dict,
      evaluation_ground_truth_data_all, evaluation_config_dict, evaluation_split,
      cfg, seq_eval_mode, 0, cat, ID_start, hw, log, dtype, device,
      differentiable_kalman_filter_config, det_id2str, force_gt_as_predicted_track,
      use_static_default_R, use_multiple_nets, show_vis, save_vis, mail, threshold_3D_IOU,
      loss_types, writer, regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight
      , q_learning_config, enable_learnable_Q=enable_learnable_Q, q_net_dict=traj_based_q_net_dict,
      trajectory_q_loss_weight=trajectory_q_loss_weight, rq_consistency_loss_weight=rq_consistency_loss_weight,
      is_training=is_training)

    # end = time.time()
    # print("并行评估时间：", end - start)
    print('evaluation_metrics_dict: ', evaluation_metrics_dict)
    # write tensorboard for this epoch
    write_tensorboard(writer, 'epoch', 0, loss_types,
                      None, None, None,
                      evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict)

  # number of optimizer.step() called
  step_index = 0
  # ===============训练==================================================================================================================================
  # training loop start with epoch_idx = 1
  for epoch_idx in range(1, num_epochs + 1):
    print('epoch_idx: ', epoch_idx)

    training_save_folder = 'training_' + training_config_dict[
      'result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % epoch_idx
    training_save_folder = os.path.join(save_dir_prefix, training_save_folder)
    training_save_dir = os.path.join(cfg.save_root, training_save_folder);
    mkdir_if_missing(training_save_dir)
    # print('training_save_dir: ', training_save_dir)
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_all_H1_epoch_0
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_0001_H1_epoch_0

    # create eval dir for each hypothesis
    eval_dir_dict = dict()
    for index in range(cfg.num_hypo):
      eval_dir_dict[index] = os.path.join(training_save_dir, 'data_%d' % index);
      mkdir_if_missing(eval_dir_dict[index])
      # print('eval_dir_dict[0]: ', eval_dir_dict[0])
    # ./results/v2v4real/multi_sensor_kalman_filter_Car_val_H1_epoch_0/data_0

    # save tracking results every epoch
    save_tracking_results = True
    training_loss_stats_avg, training_loss_stats_avg_dict, step_index = train_one_epoch(
      True, loss_types, writer, step_index,
      save_tracking_results,
      training_config_dict['seq_eval'], cav_id_list,
      training_config_dict['det_root'], eval_dir_dict, training_save_dir, training_config_dict['result_sha'],
      cfg, cat, ID_start, hw, log,
      dtype, device, differentiable_kalman_filter_config, observation_covariance_net_dict,
      optimizer, training_config_dict['len_record'], training_config_dict['detection_feature_path'],
      training_ground_truth_data_all, det_id2str,
      num_frames_backprop, num_frames_per_sub_seq, force_gt_as_predicted_track, use_static_default_R, use_multiple_nets,
      show_vis, save_vis, training_split,
      regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, clip_grad_norm,
      parameters_list, q_learning_config,delay_config,
      enable_learnable_Q=enable_learnable_Q, q_net_dict=traj_based_q_net_dict, trajectory_q_loss_weight=1,  # 新增参数
      rq_consistency_loss_weight=1)

    # save model checkpoint
    for cav_id in cav_id_list:
      model_save_file = os.path.join(training_save_dir, 'model_%s_epoch_%d.pth' % (cav_id, epoch_idx))
      torch.save(observation_covariance_net_dict[cav_id].state_dict(), model_save_file)

    # 添加保存过程噪声网络的代码
    if enable_learnable_Q:
      # for key, net in process_noise_net_dict.items():
      # process_noise_model_save_file = os.path.join(training_save_dir, 'process_noise_model_%s_epoch_%d.pth' % (key, epoch_idx))
      traj_based_q_net_model_save_file = os.path.join(training_save_dir,
                                                      'traj_based_q_net_model_epoch_%d.pth' % epoch_idx)
      torch.save(traj_based_q_net_dict[cav_id].state_dict(), traj_based_q_net_model_save_file)
    print("保存完成")
    # tracer.stop()
    # tracer.save()
    # # =====================保存CIdeep权重==========================================================================
    #     CI_trainning_save_dir = '/mnt/d/pycharm_projects/DMSTrack-master/YJQTrack/official_models_ci'
    #     os.makedirs(CI_trainning_save_dir, exist_ok=True)  # 确保目录存在
    #     ci_model_save_file = os.path.join(CI_trainning_save_dir, 'ci_weight_predictor_epoch_%d.pth' % epoch_idx)
    #     CovarianceIntersection.save_weights(ci_model_save_file)
    # # ============================================================================================================

    # run evaluation command
    if run_evaluation_every_epoch:

      # usually we do not run evaluation on training split
      # enable it only for debug
      run_evaluation_on_training_split = True
      if run_evaluation_on_training_split:
        # 'all', '0000', '0001', ...
        # this command will evaluate the current training split's tracking results
        # training_split_evaluation_command = 'python3 scripts/KITTI/evaluate.py ' + training_save_folder + ' 1 3D 0.25 %s %s' % (seq_eval_mode, training_split)
        # print('training_split_evaluation_command: ', training_split_evaluation_command)
        # returned_value = os.system(training_split_evaluation_command)
        # print('returned_value: ', returned_value)

        # instead of running another python command
        # call ab3dmot script evaluation function directly
        training_metrics_dict = evaluate(training_save_folder, mail, 1, True, False, threshold_3D_IOU, True,
                                         seq_eval_mode, training_split)
        print('training_metrics_dict: ', training_metrics_dict)
      else:
        training_metrics_dict = None

      # Evalue on evaluation split
      # if we want to evaluate on the evaluation split, need to generate the tracing results of val set and run eval command
      # inside the following evaluate() function, we will also save model checkpoints
      evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict = track_and_evaluate(
        save_dir_prefix, cav_id_list, observation_covariance_net_dict, evaluation_ground_truth_data_all,
        evaluation_config_dict, evaluation_split, cfg, seq_eval_mode, epoch_idx, cat, ID_start, hw, log, dtype, device,
        differentiable_kalman_filter_config, det_id2str, force_gt_as_predicted_track, use_static_default_R,
        use_multiple_nets, show_vis, save_vis, mail, threshold_3D_IOU, loss_types, writer,
        regression_loss_weight, association_loss_weight, det_neg_log_likelihood_loss_weight, q_learning_config,
        enable_learnable_Q=enable_learnable_Q, q_net_dict=traj_based_q_net_dict,
        trajectory_q_loss_weight=trajectory_q_loss_weight, rq_consistency_loss_weight=rq_consistency_loss_weight,
        is_training=is_training)
      print('evaluation_metrics_dict: ', evaluation_metrics_dict)

      # write tensorboard for this epoch
      write_tensorboard(writer, 'epoch', epoch_idx, loss_types,
                        training_metrics_dict, training_loss_stats_avg, training_loss_stats_avg_dict,
                        evaluation_metrics_dict, evaluation_loss_stats_avg, evaluation_loss_stats_avg_dict)

  return ID_start


def main(args):
  torch.manual_seed(0)
  np.random.seed(0)

  # load config files
  config_path = './configs/%s.yml' % args.dataset
  cfg, settings_show = Config(config_path)

  # overwrite split and detection method
  if args.split is not '': cfg.split = args.split
  if args.det_name is not '': cfg.det_name = args.det_name

  # print configs
  time_str = get_timestring()
  log = os.path.join(cfg.save_root, args.save_dir_prefix, 'log/log_%s_%s_%s.txt' % (time_str, cfg.dataset, cfg.split))
  mkdir_if_missing(log);
  log = open(log, 'w')
  print_log(args, log)
  for idx, data in enumerate(settings_show):
    print_log(data, log, display=False)

  print('config_path: ', config_path)
  print('cfg.save_root: ', cfg.save_root)
  print('args.save_dir_prefix: ', args.save_dir_prefix)

  # global ID counter used for all categories, not start from 1 for each category to prevent different
  # categories of objects have the same ID. This allows visualization of all object categories together
  # without ID conflicting, Also use 1 (not 0) as start because MOT benchmark requires positive ID
  ID_start = 1

  # deep learning parameter
  dtype = torch.float32
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('device: ', device)

  # run tracking for each category
  for cat in cfg.cat_list:
    ID_start = main_per_cat_multi_sensor_differentiable_kalman_filter(cfg, cat, log, ID_start, dtype, device,
                                                                      args.run_evaluation_every_epoch,
                                                                      args.seq_eval_mode,
                                                                      args.force_gt_as_predicted_track,
                                                                      args.use_static_default_R,
                                                                      args.use_multiple_nets,
                                                                      args.show_vis,
                                                                      args.save_vis,
                                                                      args.num_frames_backprop,
                                                                      args.num_frames_per_sub_seq,
                                                                      args.num_epochs,
                                                                      args.load_model_path,
                                                                      args.save_dir_prefix,
                                                                      args.training_split,
                                                                      args.evaluation_split,
                                                                      args.feature,
                                                                      args.regression_loss_weight,
                                                                      args.association_loss_weight,
                                                                      args.det_neg_log_likelihood_loss_weight,
                                                                      args.clip_grad_norm,
                                                                      args.use_reversed_cav_id_list,
                                                                      args.delay_config)

  # combine results for every category
  print_log('\ncombining results......', log=log)

  # MY_COMMENT: combine_trk_cat is not required since we only track 'Car'
  # combine_trk_cat(cfg.split, cfg.dataset, cfg.det_name, 'H%d' % cfg.num_hypo, cfg.num_hypo)
  # for epoch_idx in range(num_epochs):
  #  combine_trk_cat(cfg.split, cfg.dataset, cfg.det_name, 'H%d_epoch_%d' % (cfg.num_hypo, epoch_idx), cfg.num_hypo)

  print_log('\nDone!', log=log)
  log.close()


if __name__ == '__main__':
  args = parse_args()
  # 解析延迟配置
  delay_config = parse_delay_config(args.delay_config)
  print(f'Delay configuration: {delay_config}')
  main(args)
