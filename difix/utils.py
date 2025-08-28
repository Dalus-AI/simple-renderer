import torch 

def gpu_mem_report(tag=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9        # bytes of tensors the allocator knows are live
    res   = torch.cuda.memory_reserved()  / 1e9        # bytes reserved by the allocator from the driver
    peak  = torch.cuda.max_memory_allocated() / 1e9    # peak 'alloc' since last reset
    free,total = [x/1e9 for x in torch.cuda.mem_get_info()]  # driver perspective
    stats = torch.cuda.memory_stats()
    active = stats["active_bytes.all.current"] / 1e9
    inactive = stats["inactive_split_bytes.all.current"] / 1e9
    print(f"{tag} | alloc={alloc:.2f} GB  reserved={res:.2f} GB  "
          f"peak_alloc={peak:.2f} GB  free={free:.2f}/{total:.2f} GB  "
          f"active={active:.2f} GB  inactive_split={inactive:.2f} GB")