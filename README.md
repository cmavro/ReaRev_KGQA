This is the code for the EMNLP 2022 paper: "ReaRev: Adaptive Reasoning for Question Answering over Knowledge
Graphs".

We use the pre-processed data from: https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv
Download it and extract it to a folder named "data".

To run Webqsp:
python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 2 --relation_word_emb True --experiment_name Webqsp322

To run CWQ:
python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/CWQ/ --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True


We also provide some pretrained ReaRev models (ReaRev_webqsp.ckpt, ReaRev_webqsp_v2.ckpt, ReaRev_CWQ.ckpt), uploaded here https://drive.google.com/file/d/1p7eLSsSKkZQxB32mT5lMsthVP6R_3x1j/view?usp=sharing

To run try:
python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 3 --relation_word_emb True --load_experiment ReaRev_webqsp.ckpt --is_eval --name webqsp

python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 2 --relation_word_emb True --load_experiment ReaRev_webqsp_v2.ckpt --is_eval --name webqsp

python main.py ReaRev --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 --data_folder data/CWQ/ --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True --load_experiment ReaRev_CWQ.ckpt --is_eval --name cwq
