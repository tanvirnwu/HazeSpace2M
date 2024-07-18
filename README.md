<h2 align="center"><strong><a href="https://2024.acmmm.org/">Accepted in ACM MM'24, Sydney, Australia</a></strong></h2>
<h1 align="center"><strong>HazeSpace2M: A Dataset for Haze Aware Single Image Dehazing</strong></h2>

<h4 align="center">Md Tanvir Islam<sup>1</sup>, Nasir Rahim<sup>1</sup>, Saeed Anwar<sup>2</sup>, Muhammad Saqib<sup>3</sup>, Sambit Bakshi<sup>4</sup>, Khan Muhammad<sup>1</sup></h4>
<h4 align="center">| 1. Sungkyunkwan University, Republic of Korea | 2. KFUPM, KSA | 3. UTS, Australia | 4. NIT Rourkela, India |<br></h4>

----------
## HazeSpace2M Dataset
![](./assets/HazeSpace2M.jpg)
## Haze Aware Dehazing
![](./assets/proposedFramework.jpg)

## Dependencies
```
pip install -r requirements.txt
````

## Dataset Download
The subsets of the HazeSpace2M dataset are available for download from the following links:
1. Outdoor | 2. Street | 3. Farmland | 4. Satellite


## Pretrained Weights
All the pre-trained weights of the classifiers and the dehazers are available to download:
| Classifier | cloudSD | ehSD | fogSD |

## Testing
```
python inference.py --gt_folder <path_to_gt> --hazy_folder <path_to_hazy> --output_dir <output_dir> --classifier <path_to_classifier> --cloudSD <path_to_cloudSD> --ehSD <path_to_ehSD> --fogSD <path_to_fogSD>

````
_**Note:** Each variable is explained in the inference.py file._


## Use Own Classifiers
**To use your custom classifier, please follow the following steps:**
1. Write the code for your classifier architecture in the _classifier.py_ file in the _models_ folder.
2. Now define the object of your classifier in the _classification_inference_ method inside the _conditionalDehazing.py_ file under the _models_ folder.
3. Finally, define the weights of your classifier inside the _inference.py_ file
   
**To use your custom specialized dehazers, please follow the following steps:**
1. Write the code for your classifier architecture in the dehazer.py_ file in the _models_ folder.
2. Now define the object of your dehazer in the _load_model_ method inside the _helper.py_ file under the _utils_ folder.
3. Finally, define the weights of your classifier inside the _inference.py_ file

## HazeSpace2M Folder Structure
![](./assets/datasetStructure-01.jpg)
