# Copied original sampling statements, no file writes.
num_classes = graph_data.y.max().item() + 1
new_train_mask = torch.zeros_like(graph_data.train_mask)
if dataset_name == 'arxiv':
    ogb_dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    split_idx = ogb_dataset.get_idx_split()
    official_train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
    official_val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
    official_test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
    official_train_mask[split_idx['train'].to(device)] = True
    official_val_mask[split_idx['valid'].to(device)] = True
    official_test_mask[split_idx['test'].to(device)] = True
    for c in range(num_classes):
        class_mask = (graph_data.y == c) & official_train_mask
        class_indices = class_mask.nonzero().squeeze(-1)
        num_available = class_indices.shape[0]
        if num_available > 0:
            actual_shots = min(shots, num_available)
            selected_indices = torch.randperm(num_available, device=device)[:actual_shots]
            selected = class_indices[selected_indices]
            new_train_mask[selected] = True
    unused_train_nodes = official_train_mask & ~new_train_mask
    new_test_mask = official_test_mask | unused_train_nodes
    graph_data.train_mask = new_train_mask
    graph_data.val_mask = official_val_mask
    graph_data.test_mask = new_test_mask
else:
    for c in range(num_classes):
        class_indices = ((graph_data.y == c) & graph_data.train_mask).nonzero().squeeze(-1)
        num_available = class_indices.numel()
        actual_shots = min(shots, num_available)
        perm = torch.randperm(num_available, device=device)
        selected = class_indices[perm[:actual_shots]]
        new_train_mask[selected] = True
    val_indices = (~graph_data.train_mask).nonzero().squeeze(-1)
    perm = torch.randperm(val_indices.numel(), device=device)
    selected = val_indices[perm[:500]]
    new_val_mask = torch.zeros_like(graph_data.val_mask)
    new_val_mask[selected] = True
    graph_data.train_mask = new_train_mask
    graph_data.val_mask = new_val_mask
    graph_data.test_mask = ~(graph_data.train_mask | graph_data.val_mask)
