# Path to current git
base_folder_path=/path/to/Neural-Template
# Path where results should be saved
save_res_path=/path/to/train_neural_template/nt_test
# Path to dataset, mainly for test part
dataset_h5_path=/path/to/shapenet/ShapeNetDepthCore.v2.h5/splitted

#python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_class_rgbd_svr_ply_23 --max-number-gpu 2 --device-ratio 2

#python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_class_rgbd_svr_ply_23 --sample-class --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_class_RGBD_svr_23.txt --sample-class

#python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/model_epoch_1000.pth -s ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_23 --max-number-gpu 2 --device-ratio 2

#python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_23 --sample-class --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_class_RGBD_resnet50_svr_23.txt --sample-class


python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/model_epoch_1000.pth -s ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_eight --aggregate-embedding --max-number-gpu 2 --device-ratio 3 --view-use-indx-args 23 22 21 20 19 18 17 16

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_eight --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_resnet50_class_rgbd_svr_ply_aggr_eight.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/model_epoch_1000.pth -s ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_four --aggregate-embedding --max-number-gpu 2 --device-ratio 3 --view-use-indx-args 23 22 21 20

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_four --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_resnet50_class_rgbd_svr_ply_aggr_four.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/model_epoch_1000.pth -s ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_two --aggregate-embedding --max-number-gpu 2 --device-ratio 3 --view-use-indx-args 23 22

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_two --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_resnet50_class_rgbd_svr_ply_aggr_two.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder_resnet50/model_epoch_1000.pth -s ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_all --num-input-data-aggregation -1 --aggregate-embedding --max-number-gpu 2 --device-ratio 3

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_resnet50_class_rgbd_svr_ply_aggr_all --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_resnet50_class_rgbd_svr_ply_aggr_all.txt --sample-class



python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_eight --aggregate-embedding --max-number-gpu 2 --device-ratio 3 --view-use-indx-args 23 22 21 20 19 18 17 16

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_eight --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_origin_class_rgbd_svr_ply_aggr_eight.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_four --aggregate-embedding --max-number-gpu 2 --device-ratio 3 --view-use-indx-args 23 22 21 20

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_four --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_origin_class_rgbd_svr_ply_aggr_four.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_two --aggregate-embedding --max-number-gpu 2 --device-ratio 3 --view-use-indx-args 23 22

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_two --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_origin_class_rgbd_svr_ply_aggr_two.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type image --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_all --num-input-data-aggregation -1 --aggregate-embedding --max-number-gpu 2 --device-ratio 3

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_origin_class_rgbd_svr_ply_aggr_all --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_origin_model_origin_class_rgbd_svr_ply_aggr_all.txt --sample-class



python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/phase_2_model/config.py --data-path ${dataset_h5_path}/dataset_test.hdf5 --input-type voxels --obj-txt-file ${dataset_h5_path}/dataset_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/phase_2_model/model_epoch_2_310.pth -s ${save_res_path}/origin_model_class_new_data_ae_ply --max-number-gpu 2 --device-ratio 2

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_class_new_data_ae_ply --sample-class --data-path ${dataset_h5_path}/dataset_test.hdf5 --obj-txt-file ${dataset_h5_path}/dataset_test.txt --num-workers 10 -s ./debug/metrics_class_new_data_ae.txt --sample-class

python3 evaluation/generate_mesh_ply.py --config-path ${base_folder_path}/pretrain/class_pretrain/phase_2_model/config.py --data-path ${base_folder_path}/data/all_vox256_img_test.hdf5 --input-type voxels --obj-txt-file ${base_folder_path}/data/all_vox256_img_test.txt  --network-path ${base_folder_path}/pretrain/class_pretrain/phase_2_model/model_epoch_2_310.pth -s ${save_res_path}/origin_model_class_ae_ply --max-number-gpu 2 --device-ratio 2

python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_class_ae_ply --sample-class --data-path ${base_folder_path}/data/all_vox256_img_test.hdf5 --obj-txt-file ${base_folder_path}/data/all_vox256_img_test.txt --num-workers 10 -s ./debug/metrics_class_ae.txt --sample-class

#python3 evaluation/generate_mesh_ply.py --config-path ./pretrain/new_losses_run_1/image_encoder/config.py --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 --input-type image --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt  --network-path ./pretrain/new_losses_run_1/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_NEWLOSSES_svr_ply_aggregated_all_views --num-input-data-aggregation -1 --aggregate-embedding --max-number-gpu 2 --device-ratio 3

#python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_NEWLOSSES_svr_ply_aggregated_all_views --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt --num-workers 10 -s ./debug/metrics_origin_vertices_NEWLOSSES_svr_aggregated_all_views.txt

#python3 evaluation/generate_mesh_ply.py --config-path ./pretrain/new_losses_run_1/image_encoder/config.py --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 --input-type image --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt  --network-path ./pretrain/new_losses_run_1/image_encoder/model_epoch_1000.pth -s ${save_res_path}/origin_model_NEWLOSSES_svr_ply_aggregated_all_views --num-input-data-aggregation -1 --aggregate-embedding --max-number-gpu 2 --device-ratio 3

#python3 evaluation/eval.py --predicted-folder ${save_res_path}/origin_model_NEWLOSSES_svr_ply_aggregated_all_views --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt --num-workers 10 -s ./debug/metrics_origin_vertices_NEWLOSSES_svr_aggregated_all_views.txt

