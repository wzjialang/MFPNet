
# For Seasonvarying/LEVIR-CD/Google Dataset
{
  "patch_size": 256,
  "augmentation": true,
  "num_gpus": 1,
  "num_workers": 4,
  "num_channel": 3,
  "epochs": 200,
  "batch_size": 4,
  "learning_rate": 1e-4,
  "loss_function": "hybrid", # ['hybird', 'bce', 'dice', 'jaccard'], 'hybrid' means Softmax PPCE + Perceputal Loss
  "dataset_dir": "/home/bigspace/xujialang/cd_dataset/Seasonvarying/", # change to your own path
  "weight_dir": "/home/bigspace/xujialang/MFPNet_result/Seasonvarying/", # change to your own path
  "resume": "None" # Change if you want to continue your training process
}

# For Zhang dataset
{
  "patch_size": 512,
  "augmentation": true,
  "num_gpus": 1,
  "num_workers": 4,
  "num_channel": 3,
  "epochs": 200,
  "batch_size": 2,
  "learning_rate": 1e-4,
  "loss_function": "hybrid",
  "dataset_dir": "/home/bigspace/xujialang/cd_dataset/Zhang/" 
  "weight_dir": "/home/bigspace/xujialang/MFPNet_result/Zhang/",
  "resume": "None"
}
