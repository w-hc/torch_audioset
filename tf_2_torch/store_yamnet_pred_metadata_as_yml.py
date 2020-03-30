import csv
import yaml


def parse_category_meta(csv_fname):
    """Read the class name definition file and return a list of strings."""
    accu = []
    with open(csv_fname) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)   # Skip header
        for (inx, category_id, category_name) in reader:
            accu.append({'name': category_name, 'id': category_id})
    return accu


def main():
    yamnet_csv_fname = './yamnet_class_map.csv'
    meta = parse_category_meta(yamnet_csv_fname)
    with open('./yamnet_category_meta.yml', 'w') as f:
        yaml.dump(meta, f)


if __name__ == "__main__":
    main()
