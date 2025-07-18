# BWANet: A low-light image detail enhancement network with multi-scale large receptive fields


## Introduction
<details close>
<summary><b>Our Motivation and Contribution:</b></summary>
    
![CGI2025_1](https://github.com/user-attachments/assets/ce4e2bbc-5649-4fcf-b84a-6ee90f5409d0)

![CGI2025_2](https://github.com/user-attachments/assets/1b232229-290b-4d04-8f84-f4cc937c2eff)

Most existing methods fail to dynamically adapt to frequency-specific information, limiting their ability to reconstruct fine texture details. So we use bidirectional wavelet transform to decompose image features into 8 complementary frequencies and adaptively enhance them separately through attention mechanism.
![CGI2025_3](https://github.com/user-attachments/assets/8325dee5-60c0-4f38-a685-ac85ee7d0d8e)
</details>

## Network Architecture

![Figure 1](https://github.com/user-attachments/assets/6f8e6a89-3088-41ba-9236-d6eca9e3b891)

![Figure_2](https://github.com/user-attachments/assets/d4ea8e02-1351-472c-9056-c5cab11f4488)

## Dataset

You can refer to the following links to download the datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

<details close>
<summary><b> Then organize these datasets as follows: </b></summary>

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...

```

</details>


We also provide download links for LIME, NPE, MEF, DICM, and VV datasets that have no ground truth:

[Baidu Disk](https://pan.baidu.com/s/1oHg03tOfWWLp4q1R6rlzww?pwd=cyh2) (code: `cyh2`)
 or [Google Drive](https://drive.google.com/drive/folders/1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp?usp=drive_link)

 <br>

## Results

<details close>
<summary><b>Performance on LOL-v1,LOL-v2:</b></summary>
<img width="889" height="705" alt="image" src="https://github.com/user-attachments/assets/2a63fe17-3cbb-4833-86f9-7c914d1fc0c9" />

![Figure_3](https://github.com/user-attachments/assets/a4eeca83-c294-41b6-b559-4b0c1378d2e0)


</details>

<details close>
<summary><b>Performance on LIME, NPE, MEF, DICM, and VV:</b></summary>
<img width="907" height="367" alt="image" src="https://github.com/user-attachments/assets/721d54e7-b380-4bdb-8def-3abc0e6cd39c" />

![Figure_4](https://github.com/user-attachments/assets/8360982c-e46f-4092-ab9b-ff7ab1b95b5c)


</details>


## Get Started

Create Conda Environment 
```
conda create -n LLFormer python=3.7
conda activate LLFormer
conda install pytorch=1.8 torchvision=0.3 cudatoolkit=10.1 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```


### Test
You can test the pre-trained model as follows

1. Modify the paths to dataset and pre-trained mode. 
```python
# Tesing parameter 
input_dir # the path of data
result_dir # the save path of results 
weights # the weight path of the pre-trained model
```

2. Test the models for LOLv1 and LOLv2 dataset

You need to specify the data path ```input_dir```, ```result_dir```, and model path ```weight_path```. Then run
```bash
python test.py --input_dir your_data_path --result_dir your_save_path --weights weight_path

```

### Train

1. To download training and testing data

2. To train BWANet, run
```bash
python train.py -yml_path your_config_path
```
You need to modify the config for your own training environment

## Citation

If you find our work useful for your research, please cite our paper

```
@inproceedings{tang2025bwanet,
  title={BWANet: A Low-Light Image Detail Enhancement Network with Multi-Scale Large Receptive Fields},
  author={Tang, Haiyuan},
  booktitle={Computer Graphics International Conference},
  pages={?--?},
  year={2025},
  organization={Springer}
}
```

## Contact

If you have any question, please feel free to contact us via 2042497537@qq.com.

