
# run docker on 12
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
nvidia-docker run -it -v /data/haoran:/data/haoran 8e978d76921e /bin/bash

cd bert
python3 run.py -num_train_epochs 20 -batch_size 32 -save_checkpoints_steps 1000 -save_summary_steps 1000 -do_eval False -data_dir /data/haoran/aspect_based_sentiment/lc_branch/bert/zhejiang/data_ner_enforce -output_dir output -init_checkpoint /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/bert_model.ckpt -bert_config_file /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/bert_config.json -vocab_file /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/vocab.txt -device_map 1
scp haoran@10.1.129.12:/data/haoran/aspect_based_sentiment/lc_branch/bert/output/label_test.txt ./






python3 run_classifier.py -batch_size 32 -num_train_epochs 20 -max_seq_length 128 -do_train False -do_eval False -do_predict True -save_checkpoints_steps 2000 -save_summary_steps 1000 -data_dir /data/haoran/aspect_based_sentiment/lc_branch/bert/zhejiang/data_sentimental -output_dir zhejiang/output_sentiment -init_checkpoint /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/bert_model.ckpt -bert_config_file /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/bert_config.json -vocab_file /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/vocab.txt -device_map 0
scp haoran@10.1.129.12:/data/haoran/aspect_based_sentiment/lc_branch/bert/zhejiang/output_sentiment/test_results.tsv ./
scp haoran@10.1.129.12:/data/haoran/aspect_based_sentiment/lc_branch/bert/zhejiang/data_ner/category_ids.csv /Users/mo/Documents/github_projects/zhijiang/JNU/bert/zhejiang/data_ner/

python3 run_classifier.py -max_seq_length 128 -do_train False -do_eval False -do_predict True -data_dir /data/haoran/aspect_based_sentiment/lc_branch/bert/zhejiang/data_sentimental -output_dir zhejiang/output_sentiment -init_checkpoint /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/bert_model.ckpt -bert_config_file /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/bert_config.json -vocab_file /data/haoran/aspect_based_sentiment/bert/chinese_L-12_H-768_A-12/vocab.txt

docker run -p 8501:8501 --mount type=bind,source=/Users/mo/Documents/gitlab_projects/bert-bilstm-crf-ner/predict_optimizer/ner_model/,target=/models/ner_model -e MODEL_NAME=ner_model -t tensorflow/serving


scp haoran@10.1.129.12:/data/haoran/aspect_based_sentiment/lc_branch/bert/output/label2id.pkl ./