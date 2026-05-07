# python quantizer.py --category Sports_and_Outdoors --quantizer opq_pq
# python quantizer.py --category Toys_and_Games --quantizer opq_pq


nohup bash -c "python quantizer.py --category Toy_and_Games --codebook_size 256 " > quantizer.log 2>&1 &


