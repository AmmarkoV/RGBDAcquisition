def refresh_calibration(filename, calib):
    if filename is None or calib is None:
        return 0

    fp = None
    try:
        fp = open(filename, "r")
    except IOError:
        return 0

    line_length = 0
    i = 0
    category = 0
    lines_at_current_category = 0

    for line in fp:
        line = line.rstrip("\r\n")
        line_length = len(line)

        if line_length > 0:
            if line[line_length - 1] == '\n':
                line = line[:-1]
            if line[line_length - 1] == '\r':
                line = line[:-1]

        if line_length > 1:
            if line[line_length - 2] == '\n':
                line = line[:-2]
            if line[line_length - 2] == '\r':
                line = line[:-2]

        if line[0] == '%':
            lines_at_current_category = 0

        if line[0] == '%' and line[1] == 'I' and line[2] == '\0':
            category = 1
        elif line[0] == '%' and line[1] == 'D' and line[2] == '\0':
            category = 2
        elif line[0] == '%' and line[1] == 'T' and line[2] == '\0':
            category = 3
        elif line[0] == '%' and line[1] == 'R' and line[2] == '\0':
            category = 4
        elif line[0] == '%' and line[1] == 'N' and line[2] == 'F' and line[3] == '\0':
            category = 5
        elif (
            line[0] == '%'
            and line[1] == 'U'
            and line[2] == 'N'
            and line[3] == 'I'
            and line[4] == 'T'
            and line[5] == '\0'
        ):
            category = 6
        elif (
            line[0] == '%'
            and line[1] == 'R'
            and line[2] == 'T'
            and line[3] == '4'
            and line[4] == '*'
            and line[5] == '4'
            and line[6] == '\0'
        ):
            category = 7
        elif (
            line[0] == '%'
            and line[1] == 'W'
            and line[2] == 'i'
            and line[3] == 'd'
            and line[4] == 't'
            and line[5] == 'h'
            and line[6] == '\0'
        ):
            category = 8
        elif (
            line[0] == '%'
            and line[1] == 'H'
            and line[2] == 'e'
            and line[3] == 'i'
            and line[4] == 'g'
            and line[5] == 'h'
            and line[6] == 't'
            and line[7] == '\0'
        ):
            category = 9
        else:
            if category == 1:
                calib["intrinsicParametersSet"] = 1
                linesAtCurrentCategory = min(linesAtCurrentCategory, 9)
                calib["intrinsic"][linesAtCurrentCategory - 1] = float(line)
                linesAtCurrentCategory += 1
            elif category == 2:
                if linesAtCurrentCategory == 0:
                    calib["k1"] = float(line)
                elif linesAtCurrentCategory == 1:
                    calib["k2"] = float(line)
                elif linesAtCurrentCategory == 2:
                    calib["p1"] = float(line)
                elif linesAtCurrentCategory == 3:
                    calib["p2"] = float(line)
                elif linesAtCurrentCategory == 4:
                    calib["k3"] = float(line)
                linesAtCurrentCategory += 1
            elif category == 3:
                calib["extrinsicParametersSet"] = 1
                linesAtCurrentCategory = min(linesAtCurrentCategory, 3)
                calib["extrinsicTranslation"][linesAtCurrentCategory - 1] = float(line)
                linesAtCurrentCategory += 1
            elif category == 4:
                linesAtCurrentCategory = min(linesAtCurrentCategory, 3)
                calib["extrinsicRotationRodriguez"][linesAtCurrentCategory - 1] = float(line)
                linesAtCurrentCategory += 1
            elif category == 5:
                calib["nearPlane"] = float(line)
            elif category == 6:
                calib["farPlane"] = float(line)
            elif category == 7:
                linesAtCurrentCategory = min(linesAtCurrentCategory, 16)
                calib["extrinsic"][linesAtCurrentCategory - 1] = float(line)
                linesAtCurrentCategory += 1
            elif category == 8:
                calib["width"] = int(line)
            elif category == 9:
                calib["height"] = int(line)

    fp.close()
    return 1

