### Developed a 13-layer CNN with 3 fully connected layers, resulting in a total of 16 layers. The model was trained over 25 epochs, achieving approximately 82% accuracy. After training, the model was tested with a cat image, and it successfully predicted the label as "cat."
1. `git clone git@github.com:barek2k2/deep_learning_cnn.git`
2. `cd deep_learning_cnn`
3. `python3 -m pip install -r requirements.txt`
4. `python3 code_vgg_paper_cifar10.py`

The model uses CIFAR dataset with 10 classes of `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`. Also the model uses Adam optimizer(optimization algorithm) through learning rate 0.0001
