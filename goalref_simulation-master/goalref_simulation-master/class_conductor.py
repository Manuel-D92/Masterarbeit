
from __future__ import print_function
import numpy as np
from scipy.integrate import quad, dblquad
## @package class_conductor
# @brief A python class that creates several metal coil objects such as Polygonal coil,
# Elliptical coil or any combination of both shapes.
# @details This class contains several sub classes and methods. The sub classes are Polygon, Ellipse
# Ball and Puk. The methods of these classes can be used to calculate many important quantaties
#such as the induced voltage inside a conductor coil caused by a given exciter, or the magnetic flux density
# at some arbitrary point caused by a given exciter and so on.
#@author artizaa, more comments were added by ibrahiim
#@version 0.6
#@date Created on Tue Nov 17 09:59:50 2015, Modified on Tue Dec 19 10:26:50 2017
pi = np.pi
mu0 = 4*pi * 1e-7

MAXITER = 100
TOLERANCE = 1e-20
MIN_DISTANCE_TO_EXCITER = 1e-3  # [m]

########################################################################################################################

def warning(distance_to_exciter):
    """!
    This function checks, if the committed distance goes below a threshold and throws an exception, if applicable.
    
    MIN_DISTANCE_TO_EXCITER = minimal permitted distance (global variable, defined at the top of the this script)
     -> if (distance_to_exciter < MIN_DISTANCE_TO_EXCITER):
            the caller gets an AssertionError exception
     -> else:   distance_to_exciter ist returned unchanged  => function does nothing

    @param distance_to_exciter:     distance, which is compared to the global variable MIN_DISTANCE_TO_EXCITER
    @return:                        distance_to_exciter
    """

    assert (distance_to_exciter >= MIN_DISTANCE_TO_EXCITER),\
        '\x1b[1;31mWarning: Position too close to exciter wire: %f cm (< %s cm)\n' % (distance_to_exciter*100, MIN_DISTANCE_TO_EXCITER*100) +\
        '         To admit positions closer to exciter adjust parameter MIN_DISTANCE_TO_EXCITER in class_conductor\x1b[0m'

    return distance_to_exciter

########################################################################################################################

class Conductor:
    """! Basis Class

    According to it's shape, every conductor can be classified into a subclass:
        - class Polygon     --> polygonal shape, defined by list of  <I> 'support_vectors' </I>
        - class Ellipse     --> elliptic shape, defined by parameters <I> 'majorAxis' </I> and <I> 'minorAxis' </I>
        - class Ball        --> consist of 3 round coils, defined by parameter <I> 'radius' </I>
    """

    def __init__(self, windings, position, rotationMatrix, current):
        """!
        @param  windings:    number of windings
        
        @param  position:    origin of the object's coordinate system (b-frame = body frame), in relation to the inertial frame (i-frame)
        
        @param  rotationMatrix: states the orientation of the object in relation to the inertial frame; <BR>
                                NO rotation -> <B> rotationMatrix C = np.eye(3) </B>; <BR>
                                -> transform position x from b- to i-frame: <B> x_i = C * x_b + self.position </B> <BR>
                                -> inverse transform from i- to b-frame:      <B> x_b = C<SUP>T</SUP> * (x_i - self.position) </B> <BR>
        
        @param  current:    current, running through the conductor
        """
        self.windings = windings
        self.position = position
        self.orientation = rotationMatrix
        self.current = current

    # ------------------------------------------------------------------------------------------------------------------

    def set_position(self, position):
        """!
        Sets a new object position
        @param position: new position
        """
        self.position = position

    def set_orientation(self, rotationMatrix):
        """!
        Sets a new object orientation
        @param rotationMatrix: new orientation (see documentation of Constructor for detailed information)
        """
        self.orientation = rotationMatrix

    def set_current(self, current):
        """!
        Sets a new object current
        @param current: new current
        """
        self.current = current

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensity(self, point):
        #"""!
        #\vec{B}\left(P\right)=-\frac{\mu_{0}I}{4\pi}\underset{K}{\oint}\frac{\vec{r}\times d\vec{s}}{r^{3}} $
        #"""
        pass
    
    def fluxDensityApprox(self, point):
        return self.fluxDensity(point)

    def vectorPotential(self, point):
        #"""!
        #\vec{A}\left(P\right)=\frac{\mu_{0}I}{4\pi}\underset{K}{\oint}\frac{1}{r}d\vec{s}
        #"""
        pass
    
    def vectorPotentialApprox(self, point):
        return self.vectorPotential(point)

    def inducedVoltage(self, exciter, frequency, approx=False):
        #"""! u_{ind}=-N\cdot2\pi f\cdot\underset{\partial F}{\ointop}\vec{A}\cdot d\vec{s}"""
        pass
    
    def inducedVoltageApprox(self, exciter, frequency, approx=False):
        return self.inducedVoltage(exciter, frequency)

    def inducedVoltage_with_fluxDensity(self, exciter, frequency):
        #"""! u_{ind}=-N\cdot2\pi f\underset{F}{\cdot\iint}\vec{B}\cdot d\vec{F}"""
        pass


########################################################################################################################

class Polygon(Conductor):
    """!
    Subclass of Conductor.
    The shape is defined by a list of support vectors, which define the corners of the polygon.
    """

    def __init__(self, vector_list, windings, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=0, closed_loop=True):
        """!
        @param  vector_list:     list of vectors, which specify the corners of the polygon
        
        @param  windings:       number of windings
        
        @param  position:       origin of the object's coordinate system <BR>
                                <I> > default value: </I>  <B> np.array([0, 0, 0]) </B>
        
        @param  rotationMatrix: states the orientation of the object in relation to the inertial frame; <BR>
                                see documentation of Conductor for detailed information <BR>
                                <I> > default value: </I>  <B> np.eye(3) </B>
        
        @param  current:     current, running through the object <BR>
                            <I> > default value: </I>  0
                            
        @param  closed_loop:    if <B>True</B>, last point of support_vectors will be connected to first point, to obtain a closed current loop
                                if <B>False</B>, the loop will not be closed
        """
        if closed_loop == True:
            self.support_vectors = vector_list[:] + [vector_list[0]]
        else:
            self.support_vectors = vector_list[:]
        Conductor.__init__(self, windings, position, rotationMatrix, current)

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensity(self, position):
        """!
        This function evaluates the magnetic flux density B at a given position caused by a polygonal exciter wire.
        
        @param  self:       field generating object
        @param  position:   place where the flux density shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     flux density (vector with x-,y- and z-component)
        """
        b = np.array([0, 0, 0])

        # for every wire piece between to support vectors q0 an q1:
        # calculate contribution to the overall magnetic field of the polygon an add it to b
        q1 = self.support_vectors[0]
        for j in range(len(self.support_vectors)-1):
            q0 = q1
            q1 = self.support_vectors[j + 1]

            ds = q1 - q0
            r = lambda t: position - (q0 + t * ds)

            # Biot Savart's law:
            b = b + np.cross(ds, position - q0) * quad(lambda t: 1 / warning(np.linalg.norm(r(t)))** 3, 0, 1)[0]    #, epsabs=TOLERANCE

        # return flux density (equals vector):
        return self.windings * self.current * mu0 / (4*pi) * b

    # ------------------------------------------------------------------------------------------------------------------

    def vectorPotential(self, position):
        """!
        This function evaluates the magnetic vector potential A at a given position caused by a polygonal exciter wire.

        @param  self:        field generating object
        @param  position:    place where the vector potential shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     vector potential (vector with x-,y- and z-component)
        """

        # for every wire piece between to support vectors q0 an q1:
        # calculate contribution to the overall magnetic field of the polygon an add it to a
        a = np.array([0, 0, 0])
        q1 = self.support_vectors[0]
        for j in range(len(self.support_vectors)-1):
            q0 = q1
            q1 = self.support_vectors[j + 1]

            # 1/distance between point and differential wire piece
            # (given by a straight line equation):
            d = lambda t: 1 / warning(np.linalg.norm(position - q0 - t * (q1 - q0)))

            # Biot-Savart's law:
            a = a + (q1 - q0) * quad(d, 0, 1)[0]    #, epsabs=TOLERANCE

        # return vector potential (equals vector):
        return self.windings * self.current * mu0 / (4 * pi) * a

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage(self, exciter, frequency, approx=False):
        """!
        This function evaluates the induced voltage in a polygonal coil, due to the magnetic field of the given exciter.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        @param  approx:     if True -> <I>exciter.vectorPotentialApprox</I> is used instead of exact vectorPotential
        
        @return    induced voltage
        """

        # Rotation matrices (to take into account the rotation of the coil in relation to the exciter)
        c_E_t = exciter.orientation.transpose()
        c_C = self.orientation

        # The magnetic flux through the polygonal coil is calculated by integrating the vector potential of the exciter
        # along the contour of the coil. Since the contour of the coil delineates a polygon, the overall integral is
        # split into fractions from one support vector to the next.
        flux = 0
        q1 = self.support_vectors[0]
        for j in range(len(self.support_vectors)-1):
            q0 = q1  # starting point for integration
            q1 = self.support_vectors[j + 1]  # endpoint for integration

            # differential wire piece in exciter coordinates:
            ds = np.dot(np.dot(c_E_t, c_C), (q1 - q0))
            # integration point (straight line equation) in coil coordinates :
            point = lambda t: q0 + t * (q1 - q0)
            # magnetic vector potential of the exciter (in exciter coordinates):
            if approx:
                a = lambda t: exciter.vectorPotentialApprox(
                    np.dot(c_E_t, np.dot(c_C, point(t)) + self.position - exciter.position))
            else:
                a = lambda t: exciter.vectorPotential(
                    np.dot(c_E_t, np.dot(c_C, point(t)) + self.position - exciter.position))

            # magnetic flux through the coil (Stokes' theorem):
            flux = flux + quad(lambda t: np.dot(a(t), ds), 0, 1, epsabs=TOLERANCE)[0]

        # return induced voltage (law of induction):
        return -self.windings * 2*pi * frequency * flux

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltageApprox(self, exciter, frequency, approx=False):
        """!
        This function approximates the induced voltage in a polygonal coil by multiplying the exciter B-field
        component normal on the coil at its centroid with the coil's area.
        This should only be used for planar polygons enclosing a convex area. For forms satisfying these criteria
        the approximation is guaranteed to converge to the real value with decreasing coil area.

        @param      exciter:    field generating object (has to provide a method 'fluxDensity')
        @param      frequency:  frequency of the exciter field
        @param      approx:     if True -> <I>exciter.vectorPotentialApprox</I> is used instead of exact vectorPotential
        
        @return     induced voltage
        """
        
        # Rotation matrices (to take into account the rotation of the coil in relation to the exciter)
        c_E_t = exciter.orientation.transpose()
        c_C = self.orientation
        
        # Get flux density vector at center of coil
        if approx:
            bCenter = exciter.fluxDensityApprox(np.dot(c_E_t, self.position + np.dot(c_C, self.centroid) - exciter.position))
        else:
            bCenter = exciter.fluxDensity(np.dot(c_E_t, self.position + np.dot(c_C, self.centroid) - exciter.position))
        
        # Extract normal component
        nv = np.dot(c_E_t, np.dot(c_C, self.normal_vector))
        flux = np.dot(bCenter, nv) * self.area
        
        return -self.windings * 2*pi * frequency * flux

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage_with_fluxDensity(self, exciter, frequency):
        """! Works only for rectangular exciter in x-y-plane!!!"""
        x_0 = self.support_vectors[0][0]
        x_1 = self.support_vectors[2][0]
        y_0 = self.support_vectors[0][1]
        y_1 = self.support_vectors[2][1]

        # magnetic flux density of the exciter (in exciter coordinates):
        b = lambda x, y: exciter.fluxDensity(np.array([x, y, 0]) + self.position - exciter.position)

        # magnetic flux through the coil (Stokes' theorem):
        flux = dblquad(lambda y, x: b(x, y)[2], x_0, x_1, y_0, y_1, epsabs=TOLERANCE)[0]

        # law of induction:
        return -self.windings * 2*pi*frequency * flux


########################################################################################################################

class Ellipse(Conductor):
    """!
    Subclass of Conductor.
    The shape is given implicit by the parameters 'majorAxis' and 'minorAxis', which characterise an ellipse.
    The elliptic coil is located at the origin of the x-y-plane of it's coordinate system with it's major axis
    coinciding with the x-axis and the minor axis coinciding with the y-axis of the plane.

    Attributes inherited from conductor:
        - windings
        - position      > default:  np.array([0, 0, 0])
        - orientation   > default:  np.eye(3)
        - current       > default:  0

    Further Attributes:
        - majorAxis = major Axis of the ellipse (coincides with the X-axis of it's coordinate system)
        - minorAxis = minor Axis of the ellipse (coincides with the Y-axis of it's coordinate system)
    """

    def __init__(self, majorAxis, minorAxis, windings, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3),
                 current=0):
        """!
        @param majorAxis:       major Axis of the ellipse (coincides with the X-axis of it's coordinate system)
        @param minorAxis:       minor Axis of the ellipse (coincides with the Y-axis of it's coordinate system)
         
        @param  windings:       number of windings
        
        @param  position:       origin of the object's coordinate system <BR>
                                <I> > default value: </I>  <B> np.array([0, 0, 0]) </B>
        
        @param  rotationMatrix: states the orientation of the object in relation to the inertial frame; <BR>
                                see documentation of Conductor for detailed information <BR>
                                <I> > default value: </I>  <B> np.eye(3) </B>
        
        @param  current:     current, running through the object <BR>
                            <I> > default value: </I>  0
        """
        self.majorAxis = majorAxis
        self.minorAxis = minorAxis
        Conductor.__init__(self, windings, position, rotationMatrix, current)

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensity(self, position):
        """!
        This function evaluates the magnetic flux density B at a given position caused by an elliptic exciter wire.

        @param  self:       field generating object
        @param  position:   place where the flux density shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     flux density (vector with x-,y- and z-component)
        """

        # distance**3, between point and differential wire piece:
        d_3 = lambda t: warning(np.sqrt((position[0] - self.majorAxis * np.cos(t)) ** 2 +
                                        (position[1] - self.minorAxis * np.sin(t)) ** 2 + position[2] ** 2))** 3

        # some integrals needed for Biot Savart's law:
        integral_cos = quad(lambda t: self.minorAxis * np.cos(t) / d_3(t), 0, 2*pi)[0]
        integral_sin = quad(lambda t: self.majorAxis * np.sin(t) / d_3(t), 0, 2*pi)[0]
        integral_d3 = quad(lambda t: 1 / d_3(t), 0, 2*pi)[0]

        # Biot Savart's law: (split in x-, y- and z-component)
        b_x = -position[2] * integral_cos
        b_y = -position[2] * integral_sin
        b_z = position[0] * integral_cos + position[1] * integral_sin - self.majorAxis * self.minorAxis * integral_d3

        # return flux density as a vector:
        return -self.current * mu0 / (4*pi) * self.windings * np.array([b_x, b_y, b_z])

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensityApprox(self, point):
        """!
        This function approximates the magnetic flux density B at a given position caused by an elliptic exciter wire
        by modeling the coil as a magnetic dipole with a moment of coil current times coil area. 
          
        @param  self:       field generating object
        @param  position:   place where the flux density shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     approximated flux density (vector with x-,y- and z-component)
        """
        
        # Dipole moment (scalar, because only z-component exists)
        m = self.current * self.windings * pi * self.majorAxis * self.minorAxis
        # Calculate flux density at given point
        return mu0 / (4*pi) * m * (3*point[2]*point - np.dot(point,point)*np.array([0,0,1])) / np.linalg.norm(point)**5

    # ------------------------------------------------------------------------------------------------------------------

    def vectorPotential(self, position):
        """!
        This function evaluates the magnetic vector potential A at a given position caused by an elliptic exciter wire.

        @param  self:        field generating object
        @param  position:    place where the vector potential shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     vector potential (vector with x-,y- and z-component)
        """

        # distance between point and differential wire piece:
        d = lambda t: warning(np.sqrt((position[0] - self.majorAxis * np.cos(t)) ** 2 +
                                      (position[1] - self.minorAxis * np.sin(t)) ** 2 +
                                      position[2] ** 2))

        # Biot-Savart's law: (split in x- and y-component; z-component equals zero)
        a_x = quad(lambda t: -self.majorAxis * np.sin(t) / d(t), 0, 2*pi)[0]
        a_y = quad(lambda t: +self.minorAxis * np.cos(t) / d(t), 0, 2*pi)[0]

        # return vector potential as a vector:
        return self.windings * self.current * mu0 / (4*pi) * np.array([a_x, a_y, 0])

    # ------------------------------------------------------------------------------------------------------------------
    
    def vectorPotentialApprox(self, point):
        """!
        This function approximates the magnetic vector potential A at a given position caused by an elliptic exciter wire
        by modeling the coil as a magnetic dipole with a moment of coil current times coil area. 

        @param  self:        field generating object
        @param  position:    place where the vector potential shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     approximated vector potential (vector with x-,y- and z-component)
        """
        
        # Dipole moment (scalar, because only z-component exists)
        m = self.current * self.windings * pi * self.majorAxis * self.minorAxis
        # Calculate vector potential at given point
        return mu0 / (4*pi) * m / np.linalg.norm(point)**3 * np.array([-point[1], point[0], 0])

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage(self, exciter, frequency, approx=False):
        """!
        This function evaluates the induced voltage in an elliptic coil, due to the magnetic field of the given exciter.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        @param  approx:     if True -> <I>exciter.vectorPotentialApprox</I> is used instead of exact vectorPotential
        
        @return    induced voltage
        """

        # Rotation matrices (to take into account the rotation of the coil in relation to the exciter)
        c_E_t = exciter.orientation.transpose()
        c_C = self.orientation

        # differential wire piece in exciter coordinates:
        ds = lambda t: np.dot(np.dot(c_E_t, c_C), np.array([-self.majorAxis * np.sin(t),
                                                            +self.minorAxis * np.cos(t),
                                                            0]))
        # integration point (in coil coordinates):
        point = lambda t: np.array([self.majorAxis * np.cos(t),
                                    self.minorAxis * np.sin(t),
                                    0])
        # magnetic vector potential of the exciter (in exciter coordinates):
        if approx:
            a = lambda t: exciter.vectorPotentialApprox(np.dot(c_E_t, np.dot(c_C, point(t)) + self.position - exciter.position))
        else:
            a = lambda t: exciter.vectorPotential(np.dot(c_E_t, np.dot(c_C, point(t)) + self.position - exciter.position))

        # magnetic flux through the coil (Stokes' theorem):
        flux = quad(lambda t: np.dot(a(t), ds(t)), 0, 2 * pi, epsabs=TOLERANCE)[0]

        # law of induction:
        return -self.windings * 2*pi * frequency * flux

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage_with_fluxDensity(self, exciter, frequency):
        """!
        This function evaluates the induced voltage in an elliptic coil, due to the magnetic field of the given exciter.
        
        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        @param  approx:     if True -> <I>exciter.fluxDensityApprox</I> is used instead of exact fluxDensity
        
        @return    induced voltage
        """

        # Rotation matrices (to take into account the rotation of the coil in relation to the exciter)
        c_E = exciter.orientation
        c_E_t = c_E.transpose()
        c_C = self.orientation
        c_C_t = c_C.transpose()

        # Ellipse: y as function of x (needed for integration)
        y = lambda x: self.minorAxis * np.sqrt(1 - (x / self.majorAxis) ** 2)

        # magnetic flux density of the exciter (in exciter coordinates):
        b = lambda x, y: exciter.fluxDensity(np.dot(c_E_t, np.dot(c_C, np.array([x, y, 0])) + self.position - exciter.position))

        # magnetic flux through the coil (Stokes' theorem):
        flux = dblquad(lambda y, x: np.dot(c_C_t, np.dot(c_E, b(x, y)))[2], -self.majorAxis, self.majorAxis, lambda x: -y(x), lambda x: y(x), epsabs=TOLERANCE)[0]

        # law of induction:
        return -self.windings * 2*pi * frequency * flux

    # ------------------------------------------------------------------------------------------------------------------
    
    def inducedVoltageApprox(self, exciter, frequency, approx=False):
        """!
        This function approximates the induced voltage in an elliptic coil by multiplying the exciter B-field
        component normal on the coil with the coil's area.

        @param      exciter:    field generating object (has to provide a method 'fluxDensity')
        @param      frequency:  frequency of the exciter field
        @param      approx:     if True -> <I>exciter.vectorPotentialApprox</I> is used instead of exact vectorPotential
        
        @return     induced voltage
        """
        
        # Rotation matrices (to take into account the rotation of the coil in relation to the exciter)
        c_E_t = exciter.orientation.transpose()
        c_C = self.orientation
        
        # Get flux density vector at center of coil
        if approx:
            bCenter = exciter.fluxDensityApprox(np.dot(c_E_t, self.position - exciter.position))
        else:
            bCenter = exciter.fluxDensity(np.dot(c_E_t, self.position - exciter.position))
        
        # Extract normal component
        nv = np.dot(c_E_t, np.dot(c_C, np.array([0, 0, 1])))
        flux = np.dot(bCenter, nv) * pi * self.majorAxis * self.minorAxis
        
        return -self.windings * 2*pi * frequency * flux


########################################################################################################################

class Ball(Conductor):
    """!
    Subclass of Conductor.
    
    An object of class Ball consists of 3 orthogonal round coils, which are each instances of class Conductor.Ellipse. <BR>
    Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).
        - coilX     = coil with normal vector in X-direction of the ball frame
        - coilY     = coil with normal vector in Y-direction of the ball frame
        - coilZ     = coil with normal vector in Z-direction of the ball frame
        
        - c_X       = transformation matrix from coilX- to ball-frame
        - c_Y       = transformation matrix from coilY- to ball-frame
    """

    def __init__(self, radius, windings, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3),
                 current=np.array([0, 0, 0])):
        """!
        @param radius:          ball radius = radius of the three coils
        
        @param windings:        number of windings per coil
        
        @param position:        ball position = origin of the three coil coordinate systems 
        
        @param rotationMatrix:  orientation of coilZ <BR>
                                see documentation of Conductor for detailed information <BR>
                                <I> > default value: </I>  <B> np.eye(3) </B>
        
        @param current:         np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        Conductor.__init__(self, windings, position, rotationMatrix, current)
        self.c_X = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 0]])
        self.c_Y = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]])
        self.coilX = Ellipse(radius, radius, windings, position, np.dot(rotationMatrix, self.c_X), current[0])
        self.coilY = Ellipse(radius, radius, windings, position, np.dot(rotationMatrix, self.c_Y), current[1])
        self.coilZ = Ellipse(radius, radius, windings, position, rotationMatrix, current[2])

    # ------------------------------------------------------------------------------------------------------------------

    def set_position(self, position):
        self.position = position
        self.coilX.position = position
        self.coilY.position = position
        self.coilZ.position = position

    def set_orientation(self, rotationMatrix):
        self.orientation = rotationMatrix
        self.coilX.orientation = np.dot(rotationMatrix, self.c_X)
        self.coilY.orientation = np.dot(rotationMatrix, self.c_Y)
        self.coilZ.orientation = rotationMatrix

    def set_current(self, current):
        self.current = current
        self.coilX.current = current[0]
        self.coilY.current = current[1]
        self.coilZ.current = current[2]

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensity(self, position):
        """!
        This function evaluates the magnetic flux density B at a given position caused by a ball with 3 coils.
        The flux density is calculated for every single coil individually and a superposition of all 3 flux densities
        is returned.

        @param  self:       field generating object
        @param  position:   place where the flux density shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     flux density (vector with x-,y- and z-component)
        """

        return np.dot(self.c_X, self.coilX.fluxDensity(np.dot(self.c_X.transpose(), position))) + \
               np.dot(self.c_Y, self.coilY.fluxDensity(np.dot(self.c_Y.transpose(), position))) + \
               self.coilZ.fluxDensity(position)

    # ------------------------------------------------------------------------------------------------------------------

    def vectorPotential(self, position):
        """!
        This function evaluates the magnetic vector potential A at a given position caused by a ball with 3 coils.
        The vector potential is calculated for every single coil individually and a superposition of all 3 vector
        potentials is returned.

        @param  self:        field generating object
        @param  position:    place where the vector potential shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     vector potential (vector with x-,y- and z-component)
        """

        return np.dot(self.c_X, self.coilX.vectorPotential(np.dot(self.c_X.transpose(), position))) + \
               np.dot(self.c_Y, self.coilY.vectorPotential(np.dot(self.c_Y.transpose(), position))) + \
               self.coilZ.vectorPotential(position)

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage(self, exciter, frequency):
        """!
        This function evaluates the induced voltage in the 3 coils of a ball, due to the magnetic field of the given
        exciter.
        It calculates the voltage for every single coil individually and returns an array with the the 3 coil voltages.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        
        @return     induced voltage <I>= np.array([u_coilX, u_coilY, u_coilZ])</I>
        """

        return np.array([self.coilX.inducedVoltage(exciter, frequency),
                         self.coilY.inducedVoltage(exciter, frequency),
                         self.coilZ.inducedVoltage(exciter, frequency)])

    # ------------------------------------------------------------------------------------------------------------------
    
    def inducedVoltageApprox(self, exciter, frequency):
        """!
        This function approximates the induced voltage in the 3 coils of a ball by multiplying the exciter B-field
        component normal on each coil with the coil's area.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        
        @return     approximated induced voltage <I>= np.array([u_coilX, u_coilY, u_coilZ])</I>
        """
        
        return np.array([self.coilX.inducedVoltageApprox(exciter, frequency),
                         self.coilY.inducedVoltageApprox(exciter, frequency),
                         self.coilZ.inducedVoltageApprox(exciter, frequency)])


########################################################################################################################

class Puk(Conductor):
    """!
    Subclass of Conductor.
    
    An object of class ball consists of 3 orthogonal coils:
     - 1 round coil of class Conductor.Ellipse (orientation: Y)
     - 2 rectangular coils of class Conductor.Polygon (orientation: X and Z)
     Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).
        - coilX     = rectangular coil with normal vector in X-direction of the ball frame
        - coilY     = round coil with normal vector in Y-direction of the ball frame
        - coilZ     = rectangular coil with normal vector in Z-direction of the ball frame
        - c_X       = transformation matrix from coilX- in ball-frame
        - c_Y       = transformation matrix from coilY- in ball-frame
    """

    def __init__(self, radius, height, windings, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=np.array([0, 0, 0])):
        """!
        @param radius:          radius of the circular Puk's Y-coil
        @param height:          height of the Puk's rectangular coils
        @param windings:        number of windings per coil
        
        @param position:        puk position = origin of the three coil coordinate systems 
        
        @param rotationMatrix:  orientation of coilZ <BR>
                                see documentation of Conductor for detailed information <BR>
                                <I> > default value: </I>  <B> np.eye(3) </B>
        
        @param current:         np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        Conductor.__init__(self, windings, position, rotationMatrix, current)

        # Define shape of rectangular coils:
        c0 = np.array([-radius, -height/2, 0])
        c1 = np.array([radius, -height/2, 0])
        c2 = np.array([radius, height/2, 0])
        c3 = np.array([-radius, height/2, 0])
        vector_list = [c0, c1, c2, c3]
        # Rotation matrices to rotate X- and Y-coil to the inertial frame
        ## rotation around y-axis with 90 degrees anti-clockwise using the right hand rule
        ## ## the result is the new z-axis concides with the old +x-axis
        self.c_X = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 0]])
        ## rotation around x-axis with 270 degrees anti-clockwise using the right hand rule
        ## the result is the new z-axis concides with the old +y-axis
        self.c_Y = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]])
        # Create the 3 coils:
        self.coilX = Polygon(vector_list, windings, position, np.dot(rotationMatrix, self.c_X), current[0])
        self.coilY = Ellipse(radius, radius, windings, position, np.dot(rotationMatrix, self.c_Y), current[1])
        self.coilZ = Polygon(vector_list, windings, position, rotationMatrix, current[2])

    # ------------------------------------------------------------------------------------------------------------------

    def set_position(self, position):
        self.position = position
        self.coilX.position = position
        self.coilY.position = position
        self.coilZ.position = position

    def set_orientation(self, rotationMatrix):
        self.orientation = rotationMatrix
        self.coilX.orientation = np.dot(rotationMatrix, self.c_X)
        self.coilY.orientation = np.dot(rotationMatrix, self.c_Y)
        self.coilZ.orientation = rotationMatrix

    def set_current(self, current):
        self.current = current
        self.coilX.current = current[0]
        self.coilY.current = current[1]
        self.coilZ.current = current[2]

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensity(self, position):
        """!
        This function evaluates the magnetic flux density B at a given position caused by a puk with 3 coils.
        The flux density is calculated for every single coil individually and a superposition of all 3 flux densities
        is returned.

        @param  self:       field generating object
        @param  position:   place where the flux density shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     flux density (vector with x-,y- and z-component)
        """

        return np.dot(self.c_X, self.coilX.fluxDensity(np.dot(self.c_X.transpose(), position))) + \
               np.dot(self.c_Y, self.coilY.fluxDensity(np.dot(self.c_Y.transpose(), position))) + \
               self.coilZ.fluxDensity(position)

    # ------------------------------------------------------------------------------------------------------------------

    def vectorPotential(self, position):
        """!
        This function evaluates the magnetic vector potential A at a given position caused by a puk with 3 coils.
        The vector potential is calculated for every single coil individually and a superposition of all 3 vector
        potentials is returned.

        @param  self:        field generating object
        @param  position:    place where the vector potential shall be calculated <I> = np.array([x, y, z]) </I> 
        
        @return     vector potential (vector with x-,y- and z-component)
        """

        return np.dot(self.c_X, self.coilX.vectorPotential(np.dot(self.c_X.transpose(), position))) + \
               np.dot(self.c_Y, self.coilY.vectorPotential(np.dot(self.c_Y.transpose(), position))) + \
               self.coilZ.vectorPotential(position)

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage(self, exciter, frequency):
        """!
        This function evaluates the induced voltage in the 3 coils of a puk, due to the magnetic field of the given
        exciter.
        It calculates the voltage for every single coil individually and returns an array with the the 3 coil voltages.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        
        @return     induced voltage <I>= np.array([u_coilX, u_coilY, u_coilZ])</I>
        """

        return np.array([self.coilX.inducedVoltage(exciter, frequency),
                         self.coilY.inducedVoltage(exciter, frequency),
                         self.coilZ.inducedVoltage(exciter, frequency)])

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltageApprox(self, exciter, frequency):
        """!
        This function approximates the induced voltage in the 3 coils of a ball by multiplying the exciter B-field
        component normal on each coil with the coil's area.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field
        
        @return     approximated induced voltage <I>= np.array([u_coilX, u_coilY, u_coilZ])</I>
        """

        return np.array([self.coilX.inducedVoltageApprox(exciter, frequency),
                         self.coilY.inducedVoltageApprox(exciter, frequency),
                         self.coilZ.inducedVoltageApprox(exciter, frequency)])


class Wearable(Conductor):
    """!
    Subclass of Conductor.

    An object of class wearable consists of 3 orthogonal coils:

     - 3 rectangular coils of class Conductor.Polygon (orientation: X, Y and Z)
     Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).
        - coilX     = rectangular coil with normal vector in X-direction of the ball frame
        - coilY     = rectangular coil with normal vector in Y-direction of the ball frame
        - coilZ     = rectangular coil with normal vector in Z-direction of the ball frame
        - c_X       = transformation matrix from coilX- in wearable-frame
        - c_Y       = transformation matrix from coilY- in wearable-frame
    """

    def __init__(self, coil1Width, coil1Length, coil2Width, coil2Length,
                 coil3Width, coil3Length, windings, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3),
                 current=np.array([0, 0, 0])):
        """!
        @param coil1Width:          width of the Polygonal Wearable x-coil in the x-axis
        @param coil1Length:         length of the Polygonal Wearable x-coil in the y-axis
        @param coil2Width:          width of the Polygonal Wearable y-coil in the x-axis
        @param coil2Length:         length of the Polygonal Wearable y-coil in the y-axis
        @param coil3Width:          width of the Polygonal Wearable z-coil in the x-axis
        @param coil3Length:         length of the Polygonal Wearable z-coil in the y-axis
        @param windings:        number of windings per coil

        @param position:        puk position = origin of the three coil coordinate systems

        @param rotationMatrix:  orientation of coilZ <BR>
                                see documentation of Conductor for detailed information <BR>
                                <I> > default value: </I>  <B> np.eye(3) </B>

        @param current:         np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        Conductor.__init__(self, windings, position, rotationMatrix, current)

        # Define shape of rectangular coils:
        c0 = np.array([-coil1Width/2, -coil1Length/2 , 0])
        c1 = np.array([coil1Width/2, -coil1Length/2, 0])
        c2 = np.array([coil1Width/2, coil1Length/2, 0])
        c3 = np.array([-coil1Width/2, coil1Length/2, 0])
        vector_list = [c0, c1, c2, c3]
        # Rotation matrices to rotate X- and Y-coil to the inertial frame
        ## rotation around y-axis with 90 degrees anti-clockwise using the right hand rule
        ## ## the result is the new z-axis concides with the old +x-axis
        self.c_X = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 0]])

        self.coilX = Polygon(vector_list, windings, position, np.dot(rotationMatrix, self.c_X), current[0])

        # Define shape of rectangular coils:
        c0 = np.array([-coil2Width / 2, -coil2Length / 2, 0])
        c1 = np.array([coil2Width / 2, -coil2Length / 2, 0])
        c2 = np.array([coil2Width / 2, coil2Length / 2, 0])
        c3 = np.array([-coil2Width / 2, coil2Length / 2, 0])
        vector_list = [c0, c1, c2, c3]
        ## rotation around x-axis with 270 degrees anti-clockwise using the right hand rule
        ## the result is the new z-axis concides with the old +y-axis
        self.c_Y = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]])
        self.coilY = Polygon(vector_list, windings, position, np.dot(rotationMatrix, self.c_Y), current[0])

        # Define shape of rectangular coils:
        c0 = np.array([-coil3Width / 2, -coil3Length / 2, 0])
        c1 = np.array([coil3Width / 2, -coil3Length / 2, 0])
        c2 = np.array([coil3Width / 2, coil3Length / 2, 0])
        c3 = np.array([-coil3Width / 2, coil3Length / 2, 0])
        vector_list = [c0, c1, c2, c3]
        ## rotation around x-axis with 270 degrees anti-clockwise using the right hand rule
        ## the result is the new z-axis concides with the old +y-axis
        self.c_Z = np.eye(3)
        self.coilZ = Polygon(vector_list, windings, position, np.dot(rotationMatrix, self.c_Z), current[0])




    # ------------------------------------------------------------------------------------------------------------------

    def set_position(self, position):
        self.position = position
        self.coilX.position = position
        self.coilY.position = position
        self.coilZ.position = position

    def set_orientation(self, rotationMatrix):
        self.orientation = rotationMatrix
        self.coilX.orientation = np.dot(rotationMatrix, self.c_X)
        self.coilY.orientation = np.dot(rotationMatrix, self.c_Y)
        self.coilZ.orientation = rotationMatrix

    def set_current(self, current):
        self.current = current
        self.coilX.current = current[0]
        self.coilY.current = current[1]
        self.coilZ.current = current[2]

    # ------------------------------------------------------------------------------------------------------------------

    def fluxDensity(self, position):
        """!
        This function evaluates the magnetic flux density B at a given position caused by a puk with 3 coils.
        The flux density is calculated for every single coil individually and a superposition of all 3 flux densities
        is returned.

        @param  self:       field generating object
        @param  position:   place where the flux density shall be calculated <I> = np.array([x, y, z]) </I>

        @return     flux density (vector with x-,y- and z-component)
        """

        return np.dot(self.c_X, self.coilX.fluxDensity(np.dot(self.c_X.transpose(), position))) + \
               np.dot(self.c_Y, self.coilY.fluxDensity(np.dot(self.c_Y.transpose(), position))) + \
               self.coilZ.fluxDensity(position)

    # ------------------------------------------------------------------------------------------------------------------

    def vectorPotential(self, position):
        """!
        This function evaluates the magnetic vector potential A at a given position caused by a puk with 3 coils.
        The vector potential is calculated for every single coil individually and a superposition of all 3 vector
        potentials is returned.

        @param  self:        field generating object
        @param  position:    place where the vector potential shall be calculated <I> = np.array([x, y, z]) </I>

        @return     vector potential (vector with x-,y- and z-component)
        """

        return np.dot(self.c_X, self.coilX.vectorPotential(np.dot(self.c_X.transpose(), position))) + \
               np.dot(self.c_Y, self.coilY.vectorPotential(np.dot(self.c_Y.transpose(), position))) + \
               self.coilZ.vectorPotential(position)

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltage(self, exciter, frequency):
        """!
        This function evaluates the induced voltage in the 3 coils of a puk, due to the magnetic field of the given
        exciter.
        It calculates the voltage for every single coil individually and returns an array with the the 3 coil voltages.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field

        @return     induced voltage <I>= np.array([u_coilX, u_coilY, u_coilZ])</I>
        """

        return np.array([self.coilX.inducedVoltage(exciter, frequency),
                         self.coilY.inducedVoltage(exciter, frequency),
                         self.coilZ.inducedVoltage(exciter, frequency)])

    # ------------------------------------------------------------------------------------------------------------------

    def inducedVoltageApprox(self, exciter, frequency):
        """!
        This function approximates the induced voltage in the 3 coils of a ball by multiplying the exciter B-field
        component normal on each coil with the coil's area.

        @param  exciter:    field generating object (has to provide a method <I>vectorPotential</I>)
        @param  frequency:  frequency of the exciter field

        @return     approximated induced voltage <I>= np.array([u_coilX, u_coilY, u_coilZ])</I>
        """

        return np.array([self.coilX.inducedVoltageApprox(exciter, frequency),
                         self.coilY.inducedVoltageApprox(exciter, frequency),
                         self.coilZ.inducedVoltageApprox(exciter, frequency)])