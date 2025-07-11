import torch


def evaluate(model, test_dataloader, device):
    model.eval()
    correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions==labels).sum().item()
        total_images = len(test_dataloader.dataset)
        accuracy = correct/total_images
        return accuracy*100
    
def get_wrong_predictions(model, test_dataloader, device):
    wrong_images =[]
    wrong_labels=[]
    wrong_predictions=[]
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            wrong_idx = (predictions != labels).nonzero(as_tuple=True)[0]
            wrong_images.append(images[wrong_idx])
            wrong_labels.append(labels[wrong_idx])
            wrong_predictions.append(predictions[wrong_idx])

        wrong_images=torch.cat(wrong_images)
        wrong_labels=torch.cat(wrong_labels)
        wrong_predictions=torch.cat(wrong_predictions)
        return wrong_images.cpu(), wrong_labels.cpu(), wrong_predictions.cpu()

        

def get_sample_batch(test_dataloader, model, device):
    model.eval()
    sample_images, sample_labels = next(iter(test_dataloader))
    sample_images = sample_images.to(device)
    sample_labels = sample_labels.to(device)
    with torch.no_grad():
        outputs = model(sample_images)
        sample_predictions = torch.argmax(outputs, dim=1)
        return sample_images.cpu(), sample_labels.cpu(), sample_predictions.cpu()
        


    

            



