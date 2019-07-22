import torch
import torchvision
import torchvision.datasets
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F

dropout = -0.5
batchnorm = False
input_size = 28 * 28
input_channels = 1
classes = 10


# validation breaks if there is there is more than 0 valiadation images but not one of every class, disable class
# output to fix this

# reads in the datasets and converts them all to an identical format
def read_dataset(dataset, validation_size=5000, test_features_mode=False, convolution=False):
    global input_size
    global input_channels
    if dataset is 'CIFAR10':
        full_train_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
        if test_features_mode:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (1000, 49000))
            valid_set, whoops = torch.utils.data.random_split(valid_set, (1000, 48000))
            test_set = valid_set

        else:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (
                len(full_train_set) - validation_size, validation_size))
            test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True)

        input_size = 32 * 32 * 3
        input_channels = 3
    elif dataset is 'FashionMNIST':
        full_train_set = torchvision.datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
        if test_features_mode:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (1000, 59000))
            valid_set, whoops = torch.utils.data.random_split(valid_set, (1000, 58000))
            test_set = valid_set
        else:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (
                len(full_train_set) - validation_size, validation_size))
            test_set = torchvision.datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
        input_size = 28 * 28
        input_channels = 1
    elif dataset is 'MNIST':
        full_train_set = torchvision.datasets.MNIST('./data/MNIST', train=True, download=True)
        if test_features_mode:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (1000, 59000))
            valid_set, whoops = torch.utils.data.random_split(valid_set, (1000, 58000))
            test_set = valid_set

        else:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (
                len(full_train_set) - validation_size, validation_size))
            test_set = torchvision.datasets.MNIST('./data/MNIST', train=False, download=True)
        input_size = 28 * 28
        input_channels = 1

    else:
        full_train_set = torchvision.datasets.MNIST('./data/MNIST', train=True, download=True)
        if test_features_mode:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (1000, 59000))
            valid_set, whoops = torch.utils.data.random_split(valid_set, (1000, 58000))
            test_set = valid_set

        else:
            train_set, valid_set = torch.utils.data.random_split(full_train_set, (
                len(full_train_set) - validation_size, validation_size))
            test_set = torchvision.datasets.MNIST('./data/MNIST', train=False, download=True)
        input_size = 28 * 28
        input_channels = 1

    train_data, train_label = zip(*train_set)
    if not validation_size == 0:
        valid_data, valid_label = zip(*valid_set)
    else:
        valid_data, valid_label = [], []
    test_data, test_label = zip(*test_set)

    if convolution:
        train_data = rgb_dataset(train_data)
        valid_data = rgb_dataset(valid_data)
        test_data = rgb_dataset(test_data)

    custom_train_label = return_label(train_label)
    custom_valid_label = return_label(valid_label)
    custom_test_label = return_label(test_label)

    return train_data, custom_train_label, valid_data, custom_valid_label, test_data, custom_test_label


def rgb_dataset(data):
    augmented_dataset = []
    for datapoint in data:
        augmented_dataset.append(datapoint.convert('RGB'))
    return augmented_dataset


def return_label(labels):
    custom_label = []
    for label in labels:
        # if the lables are in a tensor
        try:
            custom_label.append(label.item())
        except:
            custom_label.append(label)
    return custom_label


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.train_data = x
        self.train_labels = y

    def __getitem__(self, index):
        return self.train_data[index], self.train_labels[index]

    def __len__(self):
        return len(self.train_labels)


def image_to_tensor(data):
    tensor_list = []
    for datapoint in data:
        tensor_list.append(TF.to_tensor(datapoint))
    return tensor_list


def set_seeds(index):
    random_seeds = [7, 13, 42, 69, 420]
    seed_element = random_seeds[index]
    torch.manual_seed(seed_element)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_element)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_element)


# creates a neural net on runtime according to the spesification in layers
class NeuralNetList(nn.Module):
    def __init__(self, layers):
        super(NeuralNetList, self).__init__()
        self.ReLU = nn.ReLU()
        if len(layers) > 0:
            self.fc_input = nn.Linear(input_size, layers[0])
            if batchnorm:
                self.bn_input = nn.BatchNorm1d(layers[0])
            self.linear_list = None
            if len(layers) > 1:
                self.linear_list = nn.ModuleList(
                    [nn.Linear(layers[i], layers[i + 1]) for i in range((len(layers) - 1))])
                if batchnorm:
                    self.bn_list = nn.ModuleList([nn.BatchNorm1d(layers[i + 1]) for i in range((len(layers) - 1))])
                if 0 <= dropout <= 1:
                    self.dropout = nn.Dropout(dropout)
            self.fc_out = nn.Linear(layers[-1], classes)
        # for linear regression
        else:
            self.fc_out = None
            self.linear_list = None
            self.fc_input = nn.Linear(input_size, classes)
            if batchnorm:
                self.bn_input = nn.BatchNorm1d(classes)

    def forward(self, x):
        out = self.fc_input(x)
        if batchnorm:
            out = self.bn_input(out)
        out = self.ReLU(out)
        if self.linear_list:
            for i, _ in enumerate(self.linear_list):
                out = self.linear_list[i](out)
                if batchnorm:
                    out = self.bn_list[i](out)
                if 0 <= dropout <= 1:
                    out = self.dropout(out)
                out = self.ReLU(out)

        if self.fc_out:
            out = self.fc_out(out)

        return out


# a model people can use to change it in defferent ways (you can also add convolution in this model)
class Example_neural_net(nn.Module):
    def __init__(self, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7):
        super(Example_neural_net, self).__init__()
        self.fc_input = nn.Linear(input_size, layer_1)
        self.fc1 = nn.Linear(layer_1, layer_2)
        self.fc2 = nn.Linear(layer_2, layer_3)
        self.fc3 = nn.Linear(layer_3, layer_4)
        self.fc4 = nn.Linear(layer_4, layer_5)
        self.fc5 = nn.Linear(layer_5, layer_6)
        self.fc6 = nn.Linear(layer_6, layer_7)
        self.fc_output = nn.Linear(layer_7, classes)

        self.relu = nn.ReLU()

        self.bn_input = nn.BatchNorm1d(layer_1)
        self.bn1 = nn.BatchNorm1d(layer_2)
        self.bn2 = nn.BatchNorm1d(layer_3)
        self.bn3 = nn.BatchNorm1d(layer_4)
        self.bn4 = nn.BatchNorm1d(layer_5)
        self.bn5 = nn.BatchNorm1d(layer_6)
        self.bn6 = nn.BatchNorm1d(layer_7)

    def forward(self, x):
        out = self.fc_input(x)
        out = self.bn_input(out)
        out = self.relu(out)

        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.fc6(out)
        out = self.bn6(out)
        out = self.relu(out)

        out = self.fc_output(out)
        return out


# small one layer model made for quick training
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        dense_layer_1 = 30
        self.fc1 = nn.Linear(input_size, dense_layer_1)
        self.fc_output = nn.Linear(dense_layer_1, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc_output(out)
        return out


def apply_transformations(train_data, train_label, augmentation):
    new_data = []
    new_label = []
    if augmentation['type'] is 'rotation':
        for degree in range(1, augmentation['rotation_amount'] + 1):
            new_data.append(
                transform_dataset(train_data, 'rotation_left', degrees=(degree / augmentation['rotation_degrees'])))
            new_label.append(train_label)
            new_data.append(
                transform_dataset(train_data, 'rotation_right', degrees=(degree / augmentation['rotation_degrees'])))
            new_label.append(train_label)

    if augmentation['type'] is 'scale_crop':
        for x in range(0, (augmentation['scale_crop_factor'] + 1)):
            for y in range(0, (augmentation['scale_crop_factor'] + 1)):
                new_data.append(transform_dataset(train_data, 'scale_crop', x, y, augmentation['scale_crop_factor']))
                new_label.append(train_label)

    if augmentation['type'] is 'crop_scale':
        for x in range(0, (augmentation['crop_scale_factor'] + 1)):
            for y in range(0, (augmentation['crop_scale_factor'] + 1)):
                new_data.append(transform_dataset(train_data, 'crop_scale', x, y, augmentation['crop_scale_factor']))
                new_label.append(train_label)

    if augmentation['type'] is 'duplicate':
        for _ in range(0, augmentation['duplicate_factor']):
            new_data.append(train_data)
            new_label.append(train_label)

    return new_data, new_label


def rotation_without_adding(image, degrees):
    im2 = image.rotate(degrees, expand=1)
    im = image.rotate(degrees, expand=0)

    width, height = im.size
    assert (width == height)
    new_width = width - (im2.size[0] - width)

    left = top = int((width - new_width) / 2)
    right = bottom = int((width + new_width) / 2)

    return im.crop((left, top, right, bottom))


def rotation_with_scaling(image, degrees):
    width, height = image.size
    rotated_image = rotation_without_adding(image, degrees)
    scaled_image = scale(rotated_image, width, height)
    return scaled_image


def scale(image, new_width, new_height):
    new_img = image.resize((new_width, new_height))
    return new_img


def crop(image, x1, y1, x2, y2):
    area = (x1, y1, x2, y2)
    cropped_image = image.crop(area)
    return cropped_image


# Todo add a crop scale (crop first and then scale)
def scale_crop(image, x1, y1, scaled):
    width, height = image.size
    new_width = width + scaled
    new_height = height + scaled
    large_image = scale(image, new_width, new_height)
    scale_cropped = crop(large_image, x1, y1, (x1 + width), (y1 + height))
    return scale_cropped


def transform_dataset(data, transform, degrees=0, x1=0, y1=0, scale=0):
    augmented_dataset = []
    for datapoint in data:
        if transform is 'rotation_left':
            augmented_dataset.append(rotation_with_scaling(datapoint, (-1) * degrees))
        elif transform is 'rotation_right':
            augmented_dataset.append(rotation_with_scaling(datapoint, degrees))
        elif transform is 'scale_crop':
            augmented_dataset.append(scale_crop(datapoint, x1, y1, scale))

    return augmented_dataset


# adds datasets and labels together
def data_amalgamator(orginal_data, original_label, new_data, new_label):
    full_data = []
    full_label = []
    for data, label in zip(orginal_data, original_label):
        full_data.append(data)
        full_label.append(label)
    for data_list, label_list in zip(new_data, new_label):
        try:
            for data, label in zip(data_list, label_list):
                full_data.append(data)
                full_label.append(label)

        except:
            full_data.append(data_list)
            full_label.append(label_list)

    return full_data, full_label


def evaluation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images.resize_(images.shape[0], input_size)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def classifier(convolution, test_features_mode, use_test_dataset, validation_size, dataset, custom_neural_net,
               transfer_learning,
               pretrained, resnet_version, augmentation, batch_size, learning_rate, seed, num_epochs, layers,
               device_mode):
    if device_mode == 'GPU':
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device =torch.device('cuda')
        print(device)
        torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
    dataset_name = datasets[dataset]
    set_seeds(4)

    train_data, train_label, valid_data, valid_label, test_data, test_label = read_dataset(dataset_name,
                                                                                           validation_size=validation_size,
                                                                                           test_features_mode=test_features_mode,
                                                                                           convolution=convolution)
    # augments dataset
    if not augmentation['type'] == 'None':
        new_data, new_label = apply_transformations(train_data, train_label, augmentation)
        increased_data, increased_label = data_amalgamator(train_data, train_label, new_data, new_label)
        train_data, train_label = increased_data, increased_label

    tensor_increased_train_data = image_to_tensor(train_data)
    tensor_valid_data = image_to_tensor(valid_data)
    tensor_test_data = image_to_tensor(test_data)

    train_dataset = CustomDataset(tensor_increased_train_data, train_label)
    valid_dataset = CustomDataset(tensor_valid_data, valid_label)
    test_dataset = CustomDataset(tensor_test_data, test_label)

    set_seeds(seed)

    print('learning rate = ' + str(learning_rate) + '\tseed = ' + str(seed) + '\t dataset = ' + str(
        datasets[dataset]) + ' training set size = ' + str(len(train_label)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size,
                                               shuffle=True, drop_last=False)
    if not len(valid_dataset) == 0:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size,
                                                   shuffle=True, drop_last=False)
    else:
        valid_loader = []

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size,
                                              shuffle=True, drop_last=False)

    model = None
    if test_features_mode:
        model = SmallNet().to(device)
    elif transfer_learning:
        if resnet_version == 18:
            model = torchvision.models.resnet18(pretrained=pretrained).to(device)
        elif resnet_version == 34:
            model = torchvision.models.resnet34(pretrained=pretrained).to(device)
        elif resnet_version == 50:
            model = torchvision.models.resnet50(pretrained=pretrained).to(device)
    elif custom_neural_net:
        model = Example_neural_net(1000, 1000, 1000, 1000, 1000, 1000, 1000).to(device)
    else:
        model = NeuralNetList(layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    store_loss = 0
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for _, (images, labels) in enumerate(train_loader):
            if convolution:
                images = images.to(device)
            else:
                images = images.reshape(-1, input_size).to(device)

            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            store_loss = loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch: ' + str((epoch + 1)) + '\t Loss: ' + str(store_loss) + '\t Training accuracy: ' + str(
            (correct / float(total)) * 100) + '%')

        if ((epoch + 1) % 5) == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                class_correct = list(0. for _ in range(10))
                class_total = list(0. for _ in range(10))
                for images, labels in valid_loader:
                    if convolution:
                        images = images.to(device)
                    else:
                        images = images.reshape(-1, input_size).to(device)

                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    c = (predicted == labels).squeeze()
                    for category in range(len(predicted)):
                        label = labels[category]
                        class_correct[label] += c[category].item()
                        class_total[label] += 1
                if total > 0:
                    print('Accuracy of the network on the ' + str(total) + ' validation images: {} %'.format(
                        100 * correct / float(total)))
            model.train()

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        counter = 0
        for images, labels in valid_loader:
            if convolution:
                images = images.to(device)
            else:
                images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(predicted)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                counter += 1
        print('final evaluation')
        if total > 0:
            print('Accuracy of the network on the ' + str(total) + ' validation images: {} %'.format(
                100 * correct / float(total)))

        for category in range(10):
            if class_total[category] > 0:
                print('Class: ' + str(category) + '\tAccuracy: ' + str(
                    100 * class_correct[category] / float(class_total[category])) + '\tClass size: ' + str(
                    class_total[category]))
        if total > 0:
            print('Validation\tLearning rate\tAugmentation\tseed\tepoch')
            print(str(100 * correct / float(total)) + '\t' + str(learning_rate) + '\t' +
                  str(seed) + '\t' + str(num_epochs))
    if use_test_dataset:

        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        counter = 0
        for images, labels in test_loader:
            if convolution:
                images = images.to(device)
            else:
                images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(predicted)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                counter += 1
        print('final evaluation')
        print('Accuracy of the network on the ' + str(total) + ' test images: {} %'.format(
            100 * correct / float(total)))

        for i in range(10):
            print('Class: ' + str(i) + '\tAccuracy: ' + str(
                100 * class_correct[i] / float(class_total[i])) + '\tClass size: ' + str(class_total[i]))




def populate_augmentation(type, rotation_degrees, rotation_amount,
                          scale_crop_factor, duplicate_factor):
    augmentation = {'type': type}

    if type == 'rotation':
        augmentation['rotation_degrees'] = rotation_degrees
        augmentation['rotation_amount'] = rotation_amount
    elif type == 'scale_crop':
        augmentation['scale_crop_factor'] = scale_crop_factor
    elif type == 'duplicate':
        augmentation['duplicate_factor'] = duplicate_factor
    else:
        augmentation['type'] = 'None'

    return augmentation


def run_classifier(input_convolution=False, input_test_features_mode=False,
                   input_use_test_dataset=False, input_validation_size=10000, input_dataset=0,
                   input_transfer_learning=False, input_custom_neural_net=False, input_pretrained=True,
                   input_resnet_version=18,
                   input_augmentation_type='None', input_rotation_degrees=10, input_rotation_amount=10,
                   input_scale_crop_factor=3, input_duplicate_factor=20,
                   input_batch_size=128, input_learning_rate=0.0001, input_num_epochs=50,
                   input_seed=0, input_layers=[1000, 1000, 1000], input_dropout=-1, input_batchnorm=True,
                   input_device_mode='GPU'):
    global dropout
    global batchnorm

    # does the parameter have convolution
    convolution = input_convolution
    # should it run a miniturised version of the test set
    test_features_mode = input_test_features_mode
    # Should it run the test_set
    use_test_dataset = input_use_test_dataset
    # what is the validation size
    validation_size = input_validation_size

    # Which dataset [0,1,2],['MNIST', 'FashionMNIST','CIFAR10']
    dataset = input_dataset
    # should it use a custom neural net
    custom_neural_net = input_custom_neural_net

    # should it use a pretrained model
    transfer_learning = input_transfer_learning
    if input_transfer_learning:
        convolution = True
    # should the model be trained
    pretrained = input_pretrained
    if not input_pretrained:
        convolution = True
        transfer_learning = True
    # resnet_versions = [18, 30, 54]
    resnet_version = input_resnet_version

    # inputs the types of augmentations
    # input_augmentation_type = ['None','rotate',  'scale_crop', 'duplicate']
    # None: when you do not wish to apply augmentations

    # rotate: rotates an image without adding information to the image
    # rotate degrees: the maximum of the interval of the amount of degrees the images get rotated with
    # rotateion_amount: The amount of equally spaced intervals the images will be rotated within the rotation degrees

    # scale_crop: scales the image up and then crops the image
    # scale_crop_factor: It is the amount of pixles that gets added in the scaling before it is subsiquently cropped to input sice

    # duplicate: adds a copy of all the images to the train_loader
    # duplicate_factor: the amount of coppies of the original dataset added to the loader
    augmentation = populate_augmentation(input_augmentation_type, input_rotation_degrees, input_rotation_amount,
                                         input_scale_crop_factor, input_duplicate_factor)

    # hyperparmeters
    batch_size = input_batch_size
    learning_rate = input_learning_rate
    num_epochs = input_num_epochs
    seed = input_seed

    # the following are not implemented
    # optimizer
    # loss function
    # dense layers (not sure how to add convolution due to the amount of connections to the dense layer)
    layers = input_layers
    # applied to each layer except the input layer
    dropout = input_dropout
    batchnorm = input_batchnorm

    device_mode = input_device_mode

    classifier(convolution=convolution, test_features_mode=test_features_mode, use_test_dataset=use_test_dataset,
               validation_size=validation_size, dataset=dataset, custom_neural_net=custom_neural_net,
               transfer_learning=transfer_learning,
               pretrained=pretrained, resnet_version=resnet_version, augmentation=augmentation, batch_size=batch_size,
               learning_rate=learning_rate, seed=seed, num_epochs=num_epochs, layers=layers, device_mode=device_mode)


#examples
#Quick run to make sure it is stable
run_classifier(input_test_features_mode=True, input_num_epochs=20,input_device_mode='CPU')
'''
# apply rotations to the dataset
run_classifier(input_test_features_mode=True, input_augmentation_type='rotation',input_device_mode='CPU')
#run through all the datasets
for i in range(3):
    run_classifier(input_test_features_mode=True, input_num_epochs=20,input_device_mode='CPU', input_dataset=i)
#sets the learning rate and epoch
run_classifier(input_test_features_mode=True, input_num_epochs=20, input_learning_rate=0.001 ,input_device_mode='CPU')

'''
#runs transfer learning on the dataset
run_classifier(input_transfer_learning=True, input_pretrained=True, input_resnet_version=34, input_num_epochs=10,input_batch_size=400)
#standard use of the program includes adjusting the epochs, learning rate, implementing layers, batchnorm, dropout and selects a dataset and batch size
#run_classifier(input_num_epochs=20, input_learning_rate=0.001, input_layers=[1000,500,250,100], input_batchnorm=True, input_dropout=0.1,input_dataset=0, input_batch_size=64)


