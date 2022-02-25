# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`


[comment]: <> (python train.py --do-train --run-name metaqqa --num-support 10 --num-query 100 --batch-size 1 --save-dir save/baseline-02 --train-datasets duorc,race,relation_extraction --train-dir datasets/oodomain_train)


[comment]: <> (python train.py --do-train --run-name metaqqa --num-support 10 --num-query 100 --batch-size 1 --save-dir save/baseline-02 --train-datasets duorc,race,relation_extraction --train-dir datasets/oodomain_train)



[comment]: <> (python train.py --do-train --run-name metaqqa --num-support 5 --num-query 70 --batch-size 1 --save-dir save/baseline-02 --train-datasets nat_questions,newsqa,squad --train-dir datasets/indomain_train --num-epochs 1)


[comment]: <> (python train.py --do-eval --save-dir save/baseline-02/metaqqa-16 --train-dir datasets/oodomain_train --train-datasets race,relation_extraction,duorc)




[comment]: <> ( python train.py --do-eval --save-dir baseline-02 --train-dir datasets/oodomain_train --train-datasets race,relation_extraction,duorc --eval-dir datasets/oodomain_val)