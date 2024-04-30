# CNNfashion
A CNN achieving 94% classification accuracy on the fashionminst dataset


Batch size 100, 20 epochs. AdamW optimizer, learning rate 1e-3.

First, I test for initialization dependency, which I find to be large. To do this I set
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

This ensures that each run for each seed is reproducible. Then I run over 10 initial seeds to check variance in the accuracy on the test dataset. 

In an unnormalized CNN I find that significant variance with the initial seed. 

To fix this I include batch normalization. This significantly reduces the variance in the accuracy and overall leads to an improved accuracy. 

The code shows the benchmark CNN. It achieves 94% classification accuracy on the test dataset. 

Things that I've tried that after batch normalization don't help much improving the accuracy over the benchmark model. 

1. Adding more filters in the CNNs.
2. Adding more deep fully connected layers
3. Adding dropout regularization.
4. Using average pool instead of max pool 
4. Doing batch normalization after the relus as opposed to before the relus
4. Changing the batch size to 50 or 200 as opposed to 100.
5. Putting the learning rate at 1e-2 makes things worst, at 1e-4 makes training slower, but end accuracy is similar to 1e-3. However, using an adaptive learning rate, that decreases with larger test set accuracy improves the result by a percent or so. 
6. Using SGD as opposed to Adams in the optimizer. 
7. Performing data augmentation with a random flip, 10 degree random rotation, AutoAugment with IMAGENET or CIFAR10, and RangAugment.

