import numpy as np
import class_conductor as c


class couplingFactor:
    """!
    Restore the absolute coupling factor between two coils:
        1. Compute the coupling factor from the first coil to the second:
            - Compute the flux through the primary coil
            - Compute the flux through the secondary coil
            - Divide the flux of the second coil through the first.
        2. Interconvert the two coils.
        3. Do the same as in step 1 again.
        4. Compute the absolute coupling factor.
    """
    def __init__(self, primaryCoil, secondaryCoil):
        """!
        @param primaryCoil:         coil one, with all it's attributes

        @param secondaryCoil:       coil two, with all it's attributes
        """


        self.primaryCoil = primaryCoil
        ##
        self.secondaryCoil = secondaryCoil
        self.primaryCoil_flux = 0
        self.secondaryCoil_flux = 0
        self.couplingFactor12 = 0
        self.couplingFactor21 = 0
        self.absoluteCouplingFactor = 0

        self.frequency = 1 / (2 * np.pi)

        self.secondaryCoil_current = self.secondaryCoil.current
        self.secondaryCoil.current = 0
        self.change = 0

        ## Compute flux through the primary coil:
        self.primaryCoil_flux = self.get_primaryCoil_flux(self.primaryCoil.inductance, self.primaryCoil.current)
        print('Flux through the primary coil: %s Wb' % self.primaryCoil_flux)

        ## Compute flux through the secondary coil:
        self.secondaryCoil_flux = self.secondaryCoil.inducedVoltage(self.primaryCoil, self.frequency)
        print('Flux through the secondary coil: %s Wb' % self.secondaryCoil_flux)

        ## Compute coupling factor 2,1:
        self.couplingFactor21 = self.get_couplingFactor(self.secondaryCoil_flux, self.primaryCoil_flux)
        print('Coupling factor 2,1: %s' % self.couplingFactor21)

        ## Interconvert the two coils:
        self.change = self.primaryCoil
        self.primaryCoil = self.secondaryCoil
        self.secondaryCoil = self.change

        ## Interconvert the current of the two coils:
        self.primaryCoil.current = self.secondaryCoil_current
        self.secondaryCoil.current = 0

        ## Compute flux through the primary coil:
        self.primaryCoil_flux = self.get_primaryCoil_flux(self.primaryCoil.inductance, self.primaryCoil.current)
        print('Flux through the primary coil: %s Wb' % self.primaryCoil_flux)

        ## Compute flux through the secondary coil:
        self.secondaryCoil_flux = self.secondaryCoil.inducedVoltage(self.primaryCoil, self.frequency)
        print('Flux through the secondary coil: %s Wb' % self.secondaryCoil_flux)

        ## Compute coupling factor 1,2:
        self.couplingFactor12 = self.get_couplingFactor(self.secondaryCoil_flux, self.primaryCoil_flux)
        print('Coupling factor 1,2: %s' % self.couplingFactor12)

        ## Compute absolute coupling factor:
        self.absoluteCouplingFactor = np.sqrt(self.couplingFactor21 * self.couplingFactor12)

    def get_couplingFactor(self, flux2, flux1):
        """!
        This function computes the coupling factor.

        @param  flux1:              flux of the secondary coil
        @param  flux2:              flux of the primary coil

        @return                     coupling factor between these two coils
        """
        self.flux1 = flux1
        self.flux2 = flux2

        return np.abs(self.flux2 / self.flux1)

    def get_primaryCoil_flux(self, inductance, current):
        """!
        This function computes the coupling factor.

        @param  inductance:         inductance of the primary coil
        @param  current:            current flux of the primary coil

        @return                     computed flux of the primary coil
        """
        self.inductance = inductance
        self.current = current

        return self.inductance * self.current

class PolygonExtended(c.Polygon):
    """!
       Subclass of Polygon with more atrributes.
       The shape is defined by a list of support vectors, which define the corners of the polygon.

       Attributes inherited from Conductor:
           - windings
           - position               > default:  np.array([0, 0, 0])
           - orientation            > default:  np.eye(3)
           - current                > default:  0

       Attributes inherited from Polygon:
           - vector_list            = list of vectors, which specify the corners of the polygon
           - closed_loop            = if <B>True</B>, last point of support_vectors will be connected to first point, to obtain a closed current loop
                                    if <B>False</B>, the loop will not be closed

       Further Attributes:
           - winding_width          = width of one winding
           - winding_distance       = distance between the several windings
           - radius                 = radius of the Polygon wire
           - inductance             = inductance of the Polygon
    """
    def __init__(self, vector_list, winding_width, winding_distance, radius, inductance,
                 windings, current=0, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), closed_loop=True):

        """!
        @param vector_list          list of vectors, which specify the corners of the polygon

        @winding_width              width of one winding

        @winding_distance           distance between the several windings

        @param radius:              radius of the exciter wire

        @param inductance:          inductance of the exciter wire

        @param windings:            number of windings

        @param current:             current, running through the object <BR>
                                    <I> > default value: </I>  0

        @param position:            origin of the object's coordinate system <BR>
                                    <I> > default value: </I>  <B> np.array([0, 0, 0]) </B>

        @param rotationMatrix:      states the orientation of the object in relation to the inertial frame; <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param closed_loop:         if <B>True</B>, last point of support_vectors will be connected to first point, to obtain a closed current loop
                                    if <B>False</B>, the loop will not be closed

        """
        self.vector_list = vector_list
        self.winding_width = winding_width
        self.winding_distance = winding_distance
        self.radius = radius
        self.inductance = inductance

        ## Width:
        w = 0
        ## Length:
        l = 0
        ## Height:
        h = 0

        ## Compute the circumference of the Polygon:
        for corner in self.vector_list:
            w += abs(corner[0])
            l += abs(corner[1])
            h += abs(corner[2])
        self.circumference = (w + l + h ) * 100    #[cm]

        ## For type exciter compute the inductance if not given in config:
        if self.radius != None:
            if self.inductance == None:
                self.inductance = 2 * self.circumference * (np.log(self.circumference / self.radius) + 4.0 *
                                                       self.radius / self.circumference - 1.91) * 1e-9
            else:
                self.inductance = inductance

        ## For type antenna with no radius compute the inductance if not given in config:
        elif self.radius == None:
            if self.inductance == None:
                self.inductance = 0
            else:
                self.inductance = inductance


        c.Polygon.__init__(self, vector_list=self.vector_list, windings=windings, position=position,
                         rotationMatrix=rotationMatrix, current=current, closed_loop=closed_loop)

class EllipseExtended(c.Ellipse):
    """!
    Subclass of Ellipse with more attributes.
    The shape is given implicit by the parameters 'majorAxis' and 'minorAxis', which characterise an ellipse.
    The elliptic coil is located at the origin of the x-y-plane of it's coordinate system with it's major axis
    coinciding with the x-axis and the minor axis coinciding with the y-axis of the plane.

    Attributes inherited from conductor:
        - windings
        - position                  > default:  np.array([0, 0, 0])
        - orientation               > default:  np.eye(3)
        - current                   > default:  0

    Attributes inherited from Ellipse:
        - majorAxis                 = major Axis of the ellipse (coincides with the X-axis of it's coordinate system)
        - minorAxis                 = minor Axis of the ellipse (coincides with the Y-axis of it's coordinate system)

    Further Attributes:
        - resistance                = resistance of the ellipse
        - inductance                = inductance of the ellipse, without induced inductance
    """
    def __init__(self, majorAxis, minorAxis, windings, resistance, inductance, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=0):
        """!
        @param  majorAxis:          major Axis of the ball (coincides with the X-axis of it's coordinate system)

        @param  minorAxis:          minor Axis of the ball (coincides with the Y-axis of it's coordinate system)

        @param  windings:           number of windings

        @param  resistance:         resistance of the ball

        @param  inductance:         inductance of the ball

        @param  position:           origin of the object's coordinate system <BR>
                                    <I> > default value: </I>  <B> np.array([0, 0, 0]) </B>

        @param  rotationMatrix:     states the orientation of the object in relation to the inertial frame; <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param  current:            current, running through the object <BR>
                                    <I> > default value: </I>  0
        """
        self.resistance = resistance
        self.inductance = inductance

        ## Compute the inductance if not given in config:
        if self.inductance == None:
            self.inductance = 0

        c.Ellipse.__init__(self, majorAxis, minorAxis, windings, position, rotationMatrix, current)

class BallExtended(c.Ball):
    """!
    Subclass of Ball with more attributes.
    An object of class Ball consists of 3 orthogonal round coils, which are each instances of class Conductor.Ellipse. <BR>
    Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).

    Attributes inherited from conductor:
            - windings
            - position              > default:  np.array([0, 0, 0])
            - orientation           > default:  np.eye(3)
            - current               > default:  0

    Attributes inherited from Ball:
            - coilX                 = coil with normal vector in X-direction of the ball frame
            - coilY                 = coil with normal vector in Y-direction of the ball frame
            - coilZ                 = coil with normal vector in Z-direction of the ball frame

            - c_X                   = transformation matrix from coilX- to ball-frame
            - c_Y                   = transformation matrix from coilY- to ball-frame
            - radius                = radius of the three balls

    Further Attributes:
            - resistance            = resistance of the ball
            - inductance            = inductance of the ball

    """
    def __init__(self, radius,  windings, resistance, inductance, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=np.array([0,0,0])):
        """!
        @param radius:              ball radius = radius of the three balls

        @param windings:            number of windings per coil

        @param resistance:          resistance of the ball

        @param inductance:          inductance of the ball

        @param position:            ball position = origin of the three coil coordinate systems

        @param rotationMatrix:      orientation of coilZ <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param current:             np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                    <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        self.resistance = resistance
        self.inductance = inductance

        ## Compute the inductance if not given in config:
        if self.inductance == None:
            self.inductance = 0

        c.Ball.__init__(self, radius, windings, position, rotationMatrix, current)

class PukExtended(c.Puk):
    """!
    Subclass of Puk with more attributes.

    An object of class ball consists of 3 orthogonal coils:
     - 1 round coil of class Conductor.Ellipse (orientation: Y)
     - 2 rectangular coils of class Conductor.Polygon (orientation: X and Z)
     Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).

     Attributes inherited from conductor:
            - windings
            - position              > default:  np.array([0, 0, 0])
            - orientation           > default:  np.eye(3)
            - current               > default:  0

    Attributes inherited from Puk:
            - coilX                 = coil with normal vector in X-direction of the ball frame
            - coilY                 = coil with normal vector in Y-direction of the ball frame
            - coilZ                 = coil with normal vector in Z-direction of the ball frame

            - c_X                   = transformation matrix from coilX- to ball-frame
            - c_Y                   = transformation matrix from coilY- to ball-frame

            - radius                = radius of the circular Puk's Y-coil
            - height                = height of the Puk's rectangular coils

    Further Attributes:
            - resistance            = resistance of the ball
            - inductance            = inductance of the ball
    """
    def __init__(self, radius, height, windings, resistance, inductance, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=np.array([0,0,0])):
        """!
        @param radius:              radius of the circular Puk's Y-coil

        @param height:              height of the Puk's rectangular coils

        @param windings:            number of windings per coil

        @param resistance:          resistance of the Puk

        @param inductance:          inductance of the Puk

        @param position:            Puk position = origin of the three coil coordinate systems

        @param rotationMatrix:      orientation of coilZ <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param current:             np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                    <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        self.resistance = resistance
        self.inductance = inductance

        ## Compute the inductance if not given in config:
        if self.inductance == None:
            self.inductance = 0

        c.Puk.__init__(self, radius, height, windings, position, rotationMatrix, current)