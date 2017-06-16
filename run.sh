#!/bin/sh
text_path="Data/tasks/en-valid-10k/"
data_path="Data/"

for i in "$@"
do
case $i in
    -t=*|--task=*)
    task="${i#*=}"
    shift # past argument=task
    ;;
    -tp=*|--text_path=*)
    text_path="${i#*=}"
    shift # past argument=text_path
    ;;
    -dat=*|--data_path=*)
    data_path="${i#*=}"
    shift # past argument=data_path
    ;;
    -mo=*|--mode=*)
    mode="${i#*=}"
    shift # past argument=data_path
    ;;
    *)
            # unknown option
    ;;
esac
done


#task="$1"
#text_path="$2"
#data_path="$3"
PWD=`pwd`
train_data=$PWD"/"$text_path$task"_train.txt"
valid_data=$PWD"/"$text_path$task"_valid.txt"
test_data=$PWD"/"$text_path$task"_test.txt"

echo $train_data"\n"
echo $valid_data"\n"
echo $test_data"\n"

data_processed=$PWD"/"$data_path"task_"$task"/"
model_path=$PWD"/"$data_path"task_"$task"/ModelResult/"
echo $data_processed"\n"
echo $model_path"\n"

if [ ! -d "$data_processed" ]; then
    mkdir $data_processed
    mkdir $model_path
    python Code/DataProcess.py $test_data $data_processed"vocab.json" $data_processed"test.npy"
    python Code/DataProcess.py $valid_data $data_processed"vocab.json" $data_processed"validate.npy"
    python Code/DataProcess.py $train_data $data_processed"vocab.json" $data_processed"train.npy"
fi

python Code/train.py $data_processed $mode





