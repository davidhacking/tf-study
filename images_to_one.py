# encoding=utf-8

from PIL import Image


def compare_imgs(a_imgs, b_imgs, name):
    new_img = Image.new('L', (280, 280))
    index = 0
    step = 28
    n = 10
    for i in range(0, n * step, 2 * step):
        for j in range(0, n * step, step):
            im = Image.fromarray(a_imgs[index], 'L')
            new_img.paste(im, (i, j))
            im = Image.fromarray(b_imgs[index], 'L')
            new_img.paste(im, (i + step, j))
            index += 1
    new_img.save(name)
    new_img.close()
