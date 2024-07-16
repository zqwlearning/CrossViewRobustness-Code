# How to evaluate the robustness of your model using our benchmarks？

## Fine-grained Corruption Robustness Benchmarks

### 1 Corruption Robustness Benchmark 

```python
clean_root = 'CVUSA'  # or CVACT
corruption_root = 'CVUSA-C'  # or CVACT_val-C
result_root = 'result'
severity_list = ['severity-1', 'severity-2', 'severity-3', 'severity-4', 'severity-5']
corruption_classes = ['noise', 'blur', 'weather', 'digital']
corruption_names = [
    ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    ['defocus_blur','glass_blur', 'motion_blur', 'zoom_blur'],
    ['snow', 'frost', 'fog', 'brightness', 'spatter'],
    ['contrast', 'pixelate', 'jpeg_compression']
]
```

### 2 Load Model

```python
# define your model
model = YourModel()  
 
# load your pre-trained weights for testing
path = 'your_weights.pth'  # path to your weights file
checkpoint = torch.load(path)
model.load_state_dict(checkpoint)

# define the reference branch and query branch
model_reference = model.module.reference_net  
model_query = model.module.query_net 
```

### 3 Load Reference Dataset

```python
# load reference satellite
val_reference_dataset = dataset(mode='test_reference', clean_root=clean_root, args=args)
val_reference_loader = torch.utils.data.DataLoader(val_reference_dataset, batch_size=512, shuffle=False, 					                                                        num_workers=args.workers, pin_memory=True)
```

### 4 Defining the Evaluation Function

```python
# calculate top_1, top_5, top_10, and top_1_percent
def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
    reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
    similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

    for i in range(N):
        ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                
    results = results / query_features.shape[0] * 100.
    
    return results[0], results[1], results[2], results[-1]

def validate(val_query_loader, val_reference_loader, model, args):
    # start the evaluation mode
    model_reference.eval()
    model_query.eval()
	
    # define features and labels
    query_features = np.zeros([len(val_query_loader.dataset), args.dim])
    query_labels = np.zeros([len(val_query_loader.dataset)])
    reference_features = np.zeros([len(val_reference_loader.dataset), args.dim])
    
    with torch.no_grad():
        # reference features
        for i, (images, indexes) in enumerate(val_reference_loader):
			reference_embed = model_reference(x=images, indexes=indexes)
            reference_features[indexes.cpu().numpy().astype(int), :] = reference_embed.detach().cpu().numpy()

        # query features
        for i, (images, indexes, labels) in enumerate(val_query_loader):
            query_embed = model_query(images)
            query_features[indexes.cpu().numpy(), :] = query_embed.cpu().numpy()
            query_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

        top_1, top_5, top_10, top_1_percent = accuracy(query_features, reference_features, query_labels.astype(int))
        
    return top_1, top_5, top_10, top_1_percent
```

### 5 Load Query Dataset Cyclically and Evaluate

```
for k in range(len(severity_list)):
    file = open(os.path.join(result_root, severity_list[k]), 'w+')
    for i in range(len(corruption_classes)):
        for j in range(len(corruption_names[i])):
            corruption_root = severity_list[k] + '/' + corruption_classes[i] + '/' + corruption_names[i][j]

            val_query_dataset = dataset(mode='test_query', corruption_root=corruption_root, args=args)
            val_query_loader = torch.utils.data.DataLoader(val_query_dataset, batch_size=512, shuffle=False,                                                                                num_workers=args.workers, pin_memory=True)
            top_1, top_5, top_10, top_1_percent = validate(val_query_loader, val_reference_loader, model)

            file.write(corruption_names[i][j] + ':')
            file.write(str(top_1) + ',' + str(top_5) + ',' + str(top_10) + ',' + str(top_1_percent) + '\n')
            file.flush()
    file.close()
```

## Comprehensive Corruption Robustness Benchmarks 
Since a comprehensive corruption robustness benchmark is a separate subset, simply replace the original ground query image path with the corruption robustness benchmark path.

```python
# CVUSA-C-ALL
'CVUSA/streetview/panos' ---> 'CVUSA-C-ALL'

# CVACT_val-C-ALL
'CVACT/ANU_data_small/streetview' ---> 'CVACT_val-C-ALL'

# CVACT_val-C-ALL
'CVACT/ANU_data_test/streetview' ---> 'CVACT_test-C-ALL'
```

