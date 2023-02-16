Use the following command for a test run:

```
python mufc_main.py --num_clusters=4 --num_clients=100 --data_path=celltype_processed.pkl --num_removes=10 \
                    --k_prime=4  --split=non-iid  --compare_kfed --compare_dc --client_kpp_only --verbose --update_centralized_loss
```