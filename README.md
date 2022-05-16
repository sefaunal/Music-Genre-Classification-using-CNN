# Music-Genre-Classification-using-CNN
Music Genre Classification using CNN with Python

in this project we tried to classify music genres using CNN.

#How To Use:
1. U need to run Dataset_Preparation.py to extract the neccesary values from the music files. But in order to do that u first need to download the GTZAN Dataset (if u download it from kaggle be careful however because iirc jazz0054 was broken thus causing error at this stage so u might need to remove it first if u get an error.)
2. After u get your Json file from Dataset_Preparation.py u can then run CNN_Network.py to train the network and save the output model.
3. BTW: U can skip the first two steps since the model file is already included in this repository but its up to you

Our program cannot process the .mp3 files directly since librosa causing errors thus using ffmpeg we convert .mp3 file to .wav and then make the prediction. (https://ffmpeg.org/download.html#build-windows Required files can be also found here but its already been included now)

When u re done the folder should look like this:

![image](https://user-images.githubusercontent.com/83312431/168431166-f7beb632-bcc5-49e4-933a-b105aa21c0f5.png)

Again tho DatasetV2.json is not a neccesary file since the model have already been included here.

We use Pycharm as our IDE and here is an example output from our program:

![image](https://user-images.githubusercontent.com/83312431/168431383-d6b4a1fd-8b57-4859-8ef6-f5aa6d727bc2.png)
