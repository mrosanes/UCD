import math
import bpy

############################################################################
# Initial Configurations:
# Coordinates Systems, Relative Angles between them: Rotation Parameters
# Earth location, etc.
############################################################################

# B (Magnetic Field) coordinates system (xB, yB, zB):
# If not modified -> equivalent to the absolute (Blender) coordinates system (xBlend, yBlend, zBlend)
# xB=x, yB=y, zB=z
xB = 0
yB = 0

# UCD rotation axis (zRot) regarding zB axis (y rotation)
# xRot=x1, yRot=yB=y1, zRot=z1=z2
# B coordinates system rotated around yB=yRot=y1 -> UCD star rotation axis
yRot = 10 # Angle in degrees

# UCD rotation coordinates system (x2, z2 in Plane of Sight)
# UCD rotation coordinates system rotated around zRot -> UCD rotation coordinates system
# x2=xPlaneOfSight, y2=y'=yf, z2=z1=zRot=zPlaneOfSight
z2 = 20 # Angle in degrees

# Final UCD Line of Sight coordinates system
# UCD rotation coordinates system rotated around y2=y' -> Final coordinates system
# x' in Line of Sight direction, z' in Plane of Sight
# x'=xf=xLineOfSight, y'=yf=y2, z'=zf
y2 = 30 # Angle in degrees

# earth location along the Line of Sight (LoS)
earth_location = 12
camera_offset_for_correct_view = 0.5
############################################################################
############################################################################

# Magnetic field orientation coordinates system regarding 
#   absolute (Blender) coordinates system
bpy.data.objects["ref1_B"].rotation_euler[0] = math.radians(xB)
bpy.data.objects["ref1_B"].rotation_euler[1] = math.radians(yB)

# Orientation of UCD star rotation axis (zRot) regarding zB axis (y rotation)
bpy.data.objects["ref2_UCD_rotation"].rotation_euler[1] = math.radians(yRot)

# UCD coordinates system rotated around zRot -> UCD rotation coordinates system
bpy.data.objects["ref3_planeSight"].rotation_euler[2] = math.radians(z2)

# Final UCD Line of Sight coordinates system
# UCD rotation coordinates system rotated around y2=y' -> Final coordinates system
bpy.data.objects["ref4_LineOfSight"].rotation_euler[1] = math.radians(y2)

# Earth to UCD distance along Line of sight (x' axis):
bpy.data.objects["earth"].location[0] = earth_location

# Camera Location to see the UCD from the earth
camera_location = earth_location - camera_offset_for_correct_view
bpy.data.objects["Camera"].location[0] = camera_location


