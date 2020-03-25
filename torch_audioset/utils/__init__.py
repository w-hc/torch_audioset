def mapify_iterable(iter_of_dict, field_name):
    """Convert an iterable of dicts into a big dict indexed by chosen field
    I can't think of a better name. 'Tis catchy.
    """
    acc = dict()
    for item in iter_of_dict:
        acc[item[field_name]] = item
    return acc
