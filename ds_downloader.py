import wget
import tarfile

parts = [3, 7, 4, 1, 6]

for p in parts:
    filename = f"part_{p}.tar.gz"
    WL2M_url = f"http://vision.eecs.qmul.ac.uk/datasets/WebLogo-2M/{filename}"
    wget.download(WL2M_url, filename)

    tar = tarfile.open(filename, "r:gz")
    tar.extractall("./dataset")
    tar.close()