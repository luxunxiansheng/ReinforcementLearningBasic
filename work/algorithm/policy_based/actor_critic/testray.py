import ray
ray.init()

from tqdm import tqdm

class Base:
    def __init__(self) -> None:
        print("Base")

@ray.remote
class Son(Base):
    def __init__(self) -> None:
        super().__init__()
        for index in tqdm(range(0,50)):
            print(index)


s= Son.remote()