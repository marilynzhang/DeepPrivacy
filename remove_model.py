import sys
import shutil
names = sys.argv[1:]
print(names)
answer = input("Are you sure you want to delete model \"{}\"?".format(names)).strip()

if answer != "y" and "answer" != "yes":
    exit(0)
for name in names:
    to_remove = [
        "options/{}".format(name),
        "summaries/{}".format(name),
        "checkpoints/{}".format(name),
        "generated_data/{}".format(name),
    ]
    for folder in to_remove:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            print("Folder already removed:", folder)
        print("Removed:", folder)