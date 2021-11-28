#./chain_train_script.sh -s 130 -a fcnn4
#./chain_train_script.sh -s 140 -a fcnn4a
#./chain_train_script.sh -s 150 -a fcnn4b
#./chain_train_script.sh -s 190 -a fcnn6
#./chain_train_script.sh -s 200 -a fcnn6a
#./chain_train_script.sh -s 210 -a fcnn6b

#./train_fcnn.py --arch fcnn2f --save-dir models/fcnn_run_112/ --l1 0.003 -v | tee models/fcnn_run_112/train.log
#./train_fcnn.py --arch fcnn2g --save-dir models/fcnn_run_113/ --l1 0.003 -v | tee models/fcnn_run_113/train.log
#./train_fcnn.py --arch fcnn2h --save-dir models/fcnn_run_114/ --l1 0.003 -v | tee models/fcnn_run_114/train.log
./train_fcnn.py --arch fcnn2c --save-dir models/fcnn_run_116/ --l1 0.003 -v | tee models/fcnn_run_116/train.log
#./train_fcnn.py --arch fcnn4b_d --save-dir models/fcnn_run_157/ --l1 0.003 -v | tee models/fcnn_run_157/train.log
