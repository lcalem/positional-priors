from data.pascalvoc.preprocessing.positional_prior import check_relationship


def test_box_relationship():
    '''
    '''

    # base box (xmin, ymin, xmax, ymax)
    box1 = [40, 30, 90, 70]

    # tests inclusion
    boxes_included = [
        [50, 40, 60, 60],
        [70, 31, 89, 50]
    ]

    for box2 in boxes_included:
        # add first item in the object list that we don't care for the test
        r1 = check_relationship([0] + box2, [0] + box1)
        assert r1 == 0, 'box was %s, result was %s' % (str(box2), r1)

        # check the reverse
        r2 = check_relationship([0] + box1, [0] + box2)
        assert r2 is None, 'box was %s, result was %s' % (str(box2), r2)

    # tests intersection
    boxes_inter = [
        [20, 50, 120, 100],
        [50, 65, 60, 90],
        [80, 45, 110, 55],
        [80, 60, 110, 90]
    ]

    for box2 in boxes_inter:
        r1 = check_relationship([0] + box2, [0] + box1)
        assert r1 == 1, 'box was %s, result was %s' % (str(box2), r1)

        # check the reverse
        r2 = check_relationship([0] + box1, [0] + box2)
        assert r2 == 1, 'box was %s, result was %s' % (str(box2), r2)

    # tests above
    boxes_above = [
        [10, 10, 20, 20],
        [30, 19, 60, 20],
        [50, 22, 65, 22],
        [70, 10, 100, 25]
    ]

    for box2 in boxes_above:
        r1 = check_relationship([0] + box2, [0] + box1)
        assert r1 == 2, 'box was %s, result was %s' % (str(box2), r1)

        # check the reverse
        r2 = check_relationship([0] + box1, [0] + box2)
        assert r2 == 3, 'box was %s, result was %s' % (str(box2), r2)

    # tests below
    boxes_below = [
        [130, 80, 150, 100],
        [50, 110, 70, 120],
        [80, 120, 100, 140],
        [30, 130, 50, 160]
    ]

    for box2 in boxes_below:
        r1 = check_relationship([0] + box2, [0] + box1)
        assert r1 == 3, 'box was %s, result was %s' % (str(box2), r1)

        # check the reverse
        r2 = check_relationship([0] + box1, [0] + box2)
        assert r2 == 2, 'box was %s, result was %s' % (str(box2), r2)

    print('test passed')


# python3 test_box.py
if __name__ == '__main__':
    test_box_relationship()
