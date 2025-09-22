import pydobot


device = pydobot.Dobot(port="/dev/ttyACM0")  # For Windows, replace COMXX with the actual port, e.g., COM3

home_coordinates = (233.11624145507812, -1.3117563724517822, 150.7646942138672, -0.3224027156829834)

# Move to home
def home():
    device.move_to(*home_coordinates, wait=True)


INT = (237.205078125, 40.75337219238281, 1.2212879657745361, 9.748611450195312)
def to_int():
    device.move_to(*INT, wait=True)

z_suck= -45.152862548828125
z_move = -20
x_first_row_pick = 252.2808380126953
x_first_row_drop = 255.0590362548828


# Move Blocks from Orignal Position to New Position
home()

# First Block
to_int()
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (251.57139587402344,
 -7.740585803985596,
 -42.171722412109375,
 -1.7623755931854248,
 -1.7623746395111084,
 49.472564697265625,
 72.74386596679688,
 -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (248.7977752685547,
 79.44507598876953,
 -42.7321891784668,
 17.7091064453125,
 17.7091064453125,
 50.94856643676758,
 69.8638687133789,
 -9.5367431640625e-07)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xend, yend, z_move, rstart, wait=True)

to_int()

# Second Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (251.5687713623047,
 52.991416931152344,
 -43.514408111572266,
 11.895106315612793,
 11.89510726928711,
 50.69655990600586,
 71.38036346435547,
 -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (250.12815856933594,
 140.0576171875,
 -47.03595733642578,
 29.246356964111328,
 29.246356964111328,
 56.58256149291992,
 62.74036407470703,
 -9.5367431640625e-07)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xend, yend, z_move, rstart, wait=True)

to_int()

# Third Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (309.4827575683594,
 -5.8742170333862305,
 -42.158416748046875,
 -1.0873878002166748,
 -1.0873868465423584,
 59.48506164550781,
 54.11386489868164,
 -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (307.8567199707031,
 77.20035552978516,
 -40.882293701171875,
 14.077605247497559,
 14.077606201171875,
 60.970062255859375,
 51.21586608886719,
 -9.5367431640625e-07)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xend, yend, z_move, rstart, wait=True)

to_int()

# Fourth Block
(xstart, ystart, zstart, rstart, _, _, _, _) = (310.5614929199219, 49.24079895019531, -43.2730827331543, 9.009482383728027, 9.009483337402344, 60.99256134033203, 52.691864013671875, -9.5367431640625e-07)

(xend, yend, zend, rend, _, _, _, _) = (311.10113525390625, 134.926025390625, -45.07440948486328, 23.446605682373047, 23.446605682373047, 68.06655883789062, 43.99786376953125, -9.5367431640625e-07)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xend, yend, z_move, rstart, wait=True)

to_int()


home()


# Move Blocks from New Position to Orignal Position
home()
# Fourth Block
(xstart, ystart, zstart, rstart, _, _, _, _) = (310.5614929199219, 49.24079895019531, -43.2730827331543, 9.009482383728027, 9.009483337402344, 60.99256134033203, 52.691864013671875, -9.5367431640625e-07)

(xend, yend, zend, rend, _, _, _, _) = (311.10113525390625, 134.926025390625, -45.07440948486328, 23.446605682373047, 23.446605682373047, 68.06655883789062, 43.99786376953125, -9.5367431640625e-07)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

to_int()

# Third Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (309.4827575683594,
 -5.8742170333862305,
 -42.158416748046875,
 -1.0873878002166748,
 -1.0873868465423584,
 59.48506164550781,
 54.11386489868164,
 -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (307.8567199707031,
 77.20035552978516,
 -40.882293701171875,
 14.077605247497559,
 14.077606201171875,
 60.970062255859375,
 51.21586608886719,
 -9.5367431640625e-07)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

to_int()


# Second Block
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (251.5687713623047,
 52.991416931152344,
 -43.514408111572266,
 11.895106315612793,
 11.89510726928711,
 50.69655990600586,
 71.38036346435547,
 -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (250.12815856933594,
 140.0576171875,
 -47.03595733642578,
 29.246356964111328,
 29.246356964111328,
 56.58256149291992,
 62.74036407470703,
 -9.5367431640625e-07)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)

to_int()




# First Block
to_int()
#start pos
(xstart, ystart, zstart, rstart, _, _, _, _) = (251.57139587402344,
 -7.740585803985596,
 -42.171722412109375,
 -1.7623755931854248,
 -1.7623746395111084,
 49.472564697265625,
 72.74386596679688,
 -9.5367431640625e-07)

#end pos-42.152862548828125
(xend, yend, zend, rend, _, _, _, _) = (248.7977752685547,
 79.44507598876953,
 -42.7321891784668,
 17.7091064453125,
 17.7091064453125,
 50.94856643676758,
 69.8638687133789,
 -9.5367431640625e-07)

device.move_to(xend, yend, z_move, rstart, wait=True)
device.move_to(xend, yend, z_suck, rstart, wait=True)
device.suck(True)
device.move_to(xend, yend, z_move, rstart, wait=True)

device.move_to(xstart, ystart, z_move, rstart, wait=True)
device.move_to(xstart, ystart, z_suck, rstart, wait=True)
device.suck(False)
device.move_to(xstart, ystart, z_move, rstart, wait=True)
to_int()


home()