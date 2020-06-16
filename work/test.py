import collections

People = collections.namedtuple('Person', ['name', 'age', 'gender'])

e1 = People('Asim', '15', 'F')

print(e1.__class__.__name__)

