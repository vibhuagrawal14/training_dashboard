def highlight_min_max(s, min_color="#5fba7d", max_color="#e67575"):
    """
    Highlights the max and min value in each column of the dataframe.

    :param s: the series to be processed
    :param min_color: color for highlighting the min value
    :param max_color: color for highlighting the max value
    :returns: style for applying to the series
    """
    is_max = s == s.max()
    is_min = s == s.min()
    max_mapping = [f'background-color: {max_color}' if v else '' for v in is_max]
    min_mapping = [f'background-color: {min_color}' if v else '' for v in is_min]
    return [min_mapping[i] if min_mapping[i] != '' else max_mapping[i] for i in range(len(min_mapping))]
