@echo off
echo 开始提取所有模型的推荐结果...

echo 1. 提取MF模型推荐结果
python extract_model_recommendations.py --model mf --model_path mf_addressa_checkpoint/wd_1e-05_lr_0.001_1/329_ckpt.ckpt --output_file results/mf_recommendations.json --data_path data/addressa --topk 20

echo 2. 提取MACR-MF模型推荐结果
python extract_model_recommendations.py --model macrmf --model_path mf_addressa_checkpoint/wd_1e-05_lr_0.001_0/189_ckpt.ckpt --output_file results/macrmf_recommendations.json --data_path data/addressa --c_value 40.0 --topk 20

echo 3. 提取LightGCN模型推荐结果
python extract_model_recommendations.py --model lightgcn --model_path weights/addressa/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_-60 --output_file results/lightgcn_recommendations.json --data_path data/addressa --topk 20

echo 4. 提取MACR-LightGCN模型推荐结果
python extract_model_recommendations.py --model macrlightgcn --model_path weights/addressa/LightGCN/64-64/l0.0001_r1e-05-1e-05-0.01/weights_-20 --output_file results/macrlightgcn_recommendations.json --data_path data/addressa --c_value 40.0 --topk 20

echo 所有推荐结果提取完成！ 