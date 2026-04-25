● 完整方案                                                                                
                                         
  整体流程                                                                                
                                                                                          
  数据收集脚本           训练脚本              推理加载                                   
   collect_data.py  →  train_router.py  →  serve_policy.py (加载 router.pt)
                          ↓                                                               
                    router_weights.pt                                                     
                                                                                          
  ---                                                                                     
  第一步：数据收集脚本 scripts/collect_router_data.py                                     
                                                                                          
  功能：取 ~2000 个 LiBERO 样本，跑一遍 prefix forward → 1step → 2step，保存结果
                                                                                          
  collect_router_data.py:                                   
                                                                                          
    for i in range(2000):                                                                 
      observation, actions = next(dataloader)                                             
                                                                                          
      1. _preprocess_observation(observation)                 ← 参考 serve_policy.py      
      2. embed_prefix(...)                                     ← 已有的                   
      3. language_model.forward(output_hidden_states=True)     ← 已有，拿 prefix_hidden   
         prefix_feat = prefix_hidden.mean(dim=1)               ← [1, 2048]                
                                                                                          
      4. _l1_1step(state, prefix_pad_masks, past_key_values, noise, ...)                  
         → actions_1step                                       ← [1, 10, 32]              
                                                                                          
      5. _l1_2step(state, prefix_pad_masks, past_key_values, noise, ...)                  
         → actions_2step                                       ← [1, 10, 32]              
                                                                                          
      6. diff = (actions_1step - actions_2step).abs().mean()   ← 标量                     
                                                                                          
      7. 存储 [prefix_feat.cpu().numpy(), diff.item()]                                    
                                                            
  输出文件 router_data.npz：                                                              
                                                            
  prefix_feats: (2000, 2048)    ← float32                                                 
  diffs:        (2000,)          ← float32                                                
                                                                                          
  ---                                                                                     
  第二步：训练脚本 scripts/train_router.py                                                
                                                                                          
  功能：读取数据 → 确定阈值 → 打标 → 训练 Router → 保存权重 
                                                                                          
  1. 加载数据:                                                                            
     data = np.load("router_data.npz")                                                    
     X = data["prefix_feats"]     # (2000, 2048)                                          
     diffs = data["diffs"]        # (2000,)                                               
                                                                                          
  2. 确定阈值 (百分位法):                                                                 
     threshold = np.percentile(diffs, P)   # P=60 → ~60% 样本是"简单"                     
     y = (diffs > threshold).astype(float32)   # 标签                                     
                                                                                          
  3. 构建 Router (与 __init__ 完全一致的架构):                                            
     router = nn.Sequential(                                                              
         nn.Linear(2048, 256),   # 注意: pi0 则用 1024                                    
         nn.SiLU(),                                                                       
         nn.Linear(256, 256),                                                             
         nn.SiLU(),                                                                       
         nn.Linear(256, 1),                                                               
         nn.Sigmoid(),                                                                    
     )                                                                                    
                                                                                          
  4. 训练:                                                                                
     optimizer = AdamW(router.parameters(), lr=1e-4)                                      
     for epoch in range(50):                                                              
         for batch in dataloader:                           
             pred = router(batch["feat"]).squeeze(-1)                                     
             loss = F.binary_cross_entropy(pred, batch["label"])                          
             loss.backward()                                                              
             optimizer.step()                                                             
                                                                                          
  5. 保存:                                                                                
     torch.save(router.state_dict(), "router_weights.pt")                                 
                                                                                          
  推理时加载（在 pi0_pytorch.py 或 serve 处覆写）：                                       
                                                                                          
  # checkpoint 加载后                                                                     
  router_weights = torch.load("router_weights.pt", map_location=device)                   
  model.router.load_state_dict(router_weights)                                            
  # 之后用有意义的 difficulty 替代随机输出                                                
                                                                                          
  ---                                                                                     
  关键设计问题                                                                            
                                                                                          
  Q: P 值怎么选？                                           
                                                                                          
  P=50 → 阈值 = 中位数 → 50% 走2步 → 平均 NFE = 1.5                                       
  P=60 → 阈值 = 60分位 → 40% 走2步 → 平均 NFE = 1.4                                       
  P=80 → 阈值 = 80分位 → 20% 走2步 → 平均 NFE = 1.2                                       
                                                                                          
  建议先跑数据后画 diffs                                                                  
  分布直方图，再决定。如果想先跑个快速实验，P=50（中位数）是最安全的起点，相当于把        
  random.choice([1,2]) 变成 router 选择的等价替换。                                       
                                                            
  Q: pi0 和 pi05 的 hidden_size 不同？                                                    
   
  pi05 使用 PaliGemma (Gemma 2B, width=2048)，但 pi0                                      
  用的是不同的变体。需要收集数据时确认一下 prefix_hidden 的维度。Router 第一层的
  in_features 必须匹配。                                                                  
                         