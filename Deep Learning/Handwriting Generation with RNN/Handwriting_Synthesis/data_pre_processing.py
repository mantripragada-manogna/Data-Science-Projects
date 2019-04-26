import os
import html
import pickle
import numpy as np
import xml.etree.cElementTree as ElementTree

data = []
charset = set()
file_no = 0

path = 'data'
for root, dirs, files in os.walk(path):
    # if file_no == 5:
    #     break
    for file in files:
        # if file_no == 5:
        #     break
        file_no += 1
        file_name, extension = os.path.splitext(file)
        if extension == '.xml':
            xml = ElementTree.parse(os.path.join(root, file)).getroot()
            transcription = xml.findall('Transcription')
            if not transcription:
                print('skipped')
                continue
            texts = [html.unescape(tag.get('text')) for tag in xml.findall('Transcription')[0].findall('TextLine')]

            stroke = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]
            points = [np.array([[int(p.get('x')), int(p.get('y')), 0] for p in point_tag]) for point_tag in stroke]

            strokes = []
            mid_points = []

            for point in points:
                point[-1, 2] = 1

                xmax, ymax = max(point, key=lambda x: x[0])[0], max(point, key=lambda x: x[1])[1]
                xmin, ymin = min(point, key=lambda x: x[0])[0], min(point, key=lambda x: x[1])[1]

                strokes += [point]
                mid_points += [[(xmax + xmin) / 2., (ymax + ymin) / 2.]]

            distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) for p1, p2 in zip(mid_points, mid_points[1:])]
            splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)

            lines = []
            for b, e in zip([0] + splits, splits + [len(strokes)]):
                lines += [[p for pts in strokes[b:e] for p in pts]]

            charset |= set(''.join(texts))
            data += [(texts, lines)]

# print('file name:{}; data = {}; charset = ({}) {}'.
#       format(os.path.join(root, file), len(data), len(charset), ''.join(sorted(charset))))

translation = {'<NULL>': 0}
for c in ''.join(sorted(charset)):
    translation[c] = len(translation)

dataset = []
labels = []
for texts, lines in data:
    for text, line in zip(texts, lines):
        line = np.array(line, dtype=np.float32)
        line[:, 0] = line[:, 0] - np.min(line[:, 0])
        line[:, 1] = line[:, 1] - np.min(line[:, 1])

        dataset += [line]
        labels += [list(map(lambda x: translation[x], text))]

#print(dataset)

whole_data = np.concatenate(dataset, axis=0)
std_y = np.std(whole_data[:, 1])
norm_data = []
for line in dataset:
    line[:, :2] /= std_y
    norm_data += [line]
dataset = norm_data

print('datset = {}; labels = {}'.format(len(dataset), len(labels)))

try:
    os.makedirs('data')
except FileExistsError:
    pass
np.save(os.path.join('data', 'dataset'), np.array(dataset))
np.save(os.path.join('data', 'labels'), np.array(labels))
with open(os.path.join('data', 'translation.pkl'), 'wb') as file:
    pickle.dump(translation, file)
