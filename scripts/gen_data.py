import os
import sys

def mv_dataset(gene_file, origin_root, target_root):
    """
    A temp function used to split nodule bags from gene bags
    """
    label_array = []
    name_array = [ ]
    percent_array = []
    with open(gene_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if not idx > 30:
                percent = float(line.split(",")[-1])
                percent_array.append(percent)
                label_array.append(int(line.split(",")[1]))
                name_array.append(line.split(",")[0])
    
    for idx in range(len(name_array)):
        bag_name = name_array[idx]
        label = label_array[idx]
        if label:
            os.system("""mv {} {}/gene/""".format(os.path.join(origin_root,bag_name), target_root))
        else:
            os.system("""mv {} {}/nodule/""".format(os.path.join(origin_root,bag_name), target_root))

if __name__=="__main__":
    gene_file = "../../documents/selected_genelabel.txt"
    origin_root = "/remote-home/gyf/DATA/HISMIL/10x_0.75/all/nodule"
    target_root = "/remote-home/gyf/DATA/HISMIL/10x_0.75/all/"
    mv_dataset(gene_file, origin_root, target_root)