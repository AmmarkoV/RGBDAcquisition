def bvh_to_csv(bvh_file_path_in,csv_file_path_out):
    with open(bvh_file_path_in, 'r') as file:
        lines = file.readlines()

    data = {}
    motion_start_index = None

    motionIDAssignments = list()

    # Parse Hierarchy
    for i, line in enumerate(lines):
        if 'MOTION' in line:
            motion_start_index = i + 2  # Skip the "Frames:" line
            break

        line = line.strip()

        if 'ROOT' in line:
            root_name = line.split(" ")[1].strip()
            data[root_name] = {'channels': []}
            current_joint = root_name

        elif 'JOINT' in line:
            joint_name = line.split(" ")[1].strip()
            data[joint_name] = {'channels': []}
            data[current_joint].setdefault('children', []).append(joint_name)
            current_joint = joint_name

        #elif 'End Site' in line:
        #    joint_name = 'End Site'
        #    data[current_joint].setdefault('children', []).append(joint_name)
        #    current_joint = joint_name

        elif 'CHANNELS' in line:
            splitLine = line.split(" ")
            numberOfchannels = int(splitLine[1])
            print(current_joint," has ",numberOfchannels," channels ")
            for mID in range(0,numberOfchannels): 
               motionIDAssignments.append("%s_%s" % (current_joint,splitLine[2+mID])) 


    print("Hierarchy To Motion IDS Mapping :")
    for i in range(0,len(motionIDAssignments)):
        print(i, " - ",motionIDAssignments[i])

   
    f = open(csv_file_path_out,'w')
    #------------------------------------------------------------------------  
    for column in range(len(motionIDAssignments)):
     if (column>0):
        f.write(',')
     f.write("%s"%(motionIDAssignments[column]))
    f.write('\n')
    #------------------------------------------------------------------------

    # Parse Motion
    if motion_start_index: 
        for i in range(motion_start_index, len(lines)):
          if ("Frame" not in lines[i]): #skip Frames:  / Frame Time: lines

             #Got a fresh motion line
             motionData = lines[i].split(" ")
             #Dump it to CSV
             if (len(motionData)!=len(motionIDAssignments)):
                   raise ValueError('Line %u: Mismatched number of motion values (%u) compared to our hierarchy (%u)' % (i,len(motionData),len(motionIDAssignments)) )
             #
             for column in range(len(motionIDAssignments)):
              if (column>0):
                f.write(',')
              f.write("%f"%(float(motionData[column])))
             f.write('\n')

    f.close()


if __name__ == '__main__':
      import sys
      if (len(sys.argv)>1):
        for i in range(1,len(sys.argv)):
         baseFilename = sys.argv[i]
         if (".bvh" in baseFilename):
            print("Will convert ",baseFilename)
            targetFilename = "%s.csv" % (baseFilename.rsplit(".")[0]) 
            print("To ",targetFilename)
            bvh_to_csv(baseFilename,targetFilename)
         else:
            print("Will NOT convert ",baseFilename," it does not have a .bvh extension")
      else:
         raise ValueError('Please call the utility with a single path of BVH file\n')



