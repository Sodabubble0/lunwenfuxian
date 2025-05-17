@echo off
echo 开始提取所有模型的推荐结果(简化版)...

echo 1. 提取MF模型推荐结果
python extract_mf_simple.py --model_path mf_addressa_checkpoint/wd_1e-05_lr_0.001_1/329_ckpt.ckpt --data_path data --dataset addressa --output_file results/mf_recommendations.json --topk 20

echo 2. 提取MACR-MF模型推荐结果
python extract_macrmf_simple.py --model_path mf_addressa_checkpoint/wd_1e-05_lr_0.001_0/189_ckpt.ckpt --data_path data --dataset addressa --output_file results/macrmf_recommendations.json --c_value 40.0 --topk 20

echo 3. 提取LightGCN模型推荐结果
python extract_lightgcn_simple.py --model_path weights/addressa/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_-60 --data_path data --dataset addressa --output_file results/lightgcn_recommendations.json --topk 20

echo 4. 提取MACR-LightGCN模型推荐结果
python extract_macrlightgcn_simple.py --model_path weights/addressa/LightGCN/64-64/l0.0001_r1e-05-1e-05-0.01/weights_-20 --data_path data --dataset addressa --output_file results/macrlightgcn_recommendations.json --c_value 40.0 --topk 20

echo 所有推荐结果提取完成！

echo 5. 生成图8和图9
python plot_figures.py --data_path data/addressa --results_path results --models mf,macrmf,lightgcn,macrlightgcn --model_names MF,MACR-MF,LightGCN,MACR-LightGCN --topk 20 --n_groups 4 