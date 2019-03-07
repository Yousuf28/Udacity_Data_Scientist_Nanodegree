To test the train.py file:

python train.py flowers --save_dir save_directory --arch "vgg19" --learning_rate
0.001 --hidden_units 512 --epochs 1 --gpu

create save_directory file

so predict with:

python predict.py flowers/test/28/image_05277.jpg save_directory --top_k 5
--category_names cat_to_name.json --gpu

 

python train.py flowers --arch "vgg19" --learning_rate 0.001 --hidden_units 512
--epochs 1 --gpu

create checkpoint.pth file

so preditc with:

python predict.py flowers/test/28/image_05277.jpg checkpoint.pth --top_k 5
--category_names cat_to_name.json --gpu
