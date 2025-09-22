import pydobot


device = pydobot.Dobot(port="/dev/ttyACM0")  # For Windows, replace COMXX with the actual port, e.g., COM3

home_coordinates = (233.11624145507812, -1.3117563724517822, 150.7646942138672, -0.3224027156829834)

# Move to home
def home():
    device.move_to(*home_coordinates, wait=True)


INT = (237.205078125, 40.75337219238281, 1.2212879657745361, 9.748611450195312)
def to_int():
    device.move_to(*INT, wait=True)


z_suck= -20
z_move = 15
x_first_row_pick = 252.2808380126953
x_first_row_drop = 255.0590362548828

# Move Blocks from Orignal Position to New Position
home()

# First Block(311.10113525390625, 134.926025390625, -45.07440948486328, 23.446605682373047, 23.446605682373047, 68.06655883789062, 43.99786376953125, -9.5367431640625e-07)

to_int()
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (254.315673828125, -20.530563354492188, -14.178945541381836, -4.615403175354004, -4.6154022216796875, 39.248565673828125, 63.145362854003906, -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (251.58465576171875, 66.56598663330078, -23.257755279541016, 14.82010555267334, 14.820106506347656, 43.568565368652344, 64.65286254882812, -9.5367431640625e-07)
device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)

# to_int()

# Second Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (253.10755920410156, 36.83464431762695, -17.6192684173584, 8.280104637145996, 8.280105590820312, 40.70656204223633, 64.1578598022461, -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (257.3385009765625, 123.53132629394531, -21.634689331054688, 25.642608642578125, 25.642608642578125, 47.942562103271484, 56.98036575317383, -9.5367431640625e-07)

device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)


# to_int()

#  Third Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (316.8089294433594, -17.41505241394043, -17.285524368286133, -3.21589732170105, -3.2158963680267334, 52.33006286621094, 48.35386657714844, -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (317.0660400390625, 64.5036392211914, -20.18846321105957, 11.900729179382324, 11.90073013305664, 53.792564392089844, 48.33586502075195, -9.5367431640625e-07)

device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)


# to_int()

# Fourth Block
(xstart, ystart, zstart, rstart, _, _, _, _) = (316.8089294433594, 33.82275390625, -24.353979110717773, 6.093857288360596, 6.0938568115234375, 56.45656204223633, 47.395362854003906, 6.258487701416016e-07)
                                                
(xend, yend, zend, rend, _, _, _, _) = (317.0660400390625, 123.69127655029297, -23.44489288330078, 21.311355590820312, 21.311355590820312, 62.4865608215332, 39.65986251831055, -9.5367431640625e-07)
    
device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)

to_int()


home()

# Move Blocks from New Position to Orignal Position
home()

# First Block(311.10113525390625, 134.926025390625, -45.07440948486328, 23.446605682373047, 23.446605682373047, 68.06655883789062, 43.99786376953125, -9.5367431640625e-07)

to_int()
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (254.315673828125, -20.530563354492188, -14.178945541381836, -4.615403175354004, -4.6154022216796875, 39.248565673828125, 63.145362854003906, -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (251.58465576171875, 66.56598663330078, -23.257755279541016, 14.82010555267334, 14.820106506347656, 43.568565368652344, 64.65286254882812, -9.5367431640625e-07)

device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)



# to_int()

# Second Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (253.10755920410156, 36.83464431762695, -17.6192684173584, 8.280104637145996, 8.280105590820312, 40.70656204223633, 64.1578598022461, -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (257.3385009765625, 123.53132629394531, -21.634689331054688, 25.642608642578125, 25.642608642578125, 47.942562103271484, 56.98036575317383, -9.5367431640625e-07)

device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)


# to_int()

# Third Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (316.8089294433594, -17.41505241394043, -17.285524368286133, -3.21589732170105, -3.2158963680267334, 52.33006286621094, 48.35386657714844, -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (317.0660400390625, 64.5036392211914, -20.18846321105957, 11.900729179382324, 11.90073013305664, 53.792564392089844, 48.33586502075195, -9.5367431640625e-07)

device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

# to_int()

# Fourth Block
(xstart, ystart, zstart, rstart, _, _, _, _) = (316.8089294433594, 33.82275390625, -24.353979110717773, 6.093857288360596, 6.0938568115234375, 56.45656204223633, 47.395362854003906, 6.258487701416016e-07)
                                                
(xend, yend, zend, rend, _, _, _, _) = (317.0660400390625, 123.69127655029297, -23.44489288330078, 21.311355590820312, 21.311355590820312, 62.4865608215332, 39.65986251831055, -9.5367431640625e-07)
    
device.grip(False)
device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.grip(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.grip(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

to_int()


home()