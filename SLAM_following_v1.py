import time # used for time.sleep
from vibes import *
from codac import *
from codac import SepPolarXY
import math
import numpy as np

# Based on previous work from Simon Rouhou, Luc Jaulin and others from Codac team (codac.io)

# In particular :
# https://github.com/codac-team/codac/tree/master/doc/doc/tutorial/07-data-association
# Simon Rohou, Benoît Desrochers, Luc Jaulin
# ICRA 2020, Paris

class myCtc(Ctc):

  def __init__(self, map_):
    Ctc.__init__(self, 2)
    self.map = map_

  def contract(self, x):
    envelope = IntervalVector(2,Interval.EMPTY_SET)
    for m in self.map:
      envelope |= x & m
    x = envelope
    return x

def control(a,b,x):
    # Defines controller based on sailboat method (without considering wind effect on navigation)
    v=(b-a)/np.linalg.norm(b-a)
    w=x[0:2]-a # NOT interval formalism at first (further dev on generalization )
    e=v[0]*w[1] - v[1]*w[0] # det related to distance 
    phi=math.atan2(b[1]-a[1],b[0]-a[0])

    # Inspired by tank line exercise, see video if needed for more explanations
    theta_d=phi-math.atan(e)
    a_h=2 # Max control heading 
    u=a_h*math.atan(math.tan((theta_d-x[2])/2)) # sawtooth and atan saturator for security issues
    return u

def ref_evolution(x_ref,u,dt): # supposed unfirom speed of 1 (to be tuned)
  rng=np.random.default_rng()
  return [x_ref[0] + dt*(1*cos(x_ref[2]) + 0.1*rng.standard_normal()),
          x_ref[1] + dt*(1*sin(x_ref[2]) + 0.1*rng.standard_normal()),
          x_ref[2] + dt*(u + 5*math.pi/180*rng.standard_normal())]

# Objective : line to follow (supposed horizontal here)
a,b=np.array([0,0]),np.array([10,0])

# =========== CREATING DATA ===========

dt,tmax = 0.05,10 # tube timestep
tdomain = Interval(0,tmax)

# Initial pose x0
x0 = (0,0,+math.pi/3)
# System input
u0,u_values = 0,{}
x_ref=TrajectoryVector(3)

u_values[0]=u0
u_ref=Trajectory(u_values)
x_ref.set(x0,0)

# =========== CUSTOM-BUILT CONTRACTORS ===========

ctc_plus = CtcFunction(Function("a", "b", "c", "a-(b+c)")) # algebraic constraint a=b+c
ctc_minus = CtcFunction(Function("a", "b", "c", "a-(b-c)")) # algebraic constraint a=b-c

ctc_f = CtcFunction(
  Function("v[3]", "x[3]", "u",
           "(v[0]-1*cos(x[2]) ; v[1]-1*sin(x[2]) ; v[2]-u)"))

x_all=TubeVector(tdomain,dt,IntervalVector(3))
u_all=Tube(tdomain,dt,Interval())

x_all.set(IntervalVector(x0),0)
u_all.set(Interval(u0),0)

beginDrawing()
fig_map = VIBesFigMap("Online-SLAM Line Following")
fig_map.set_properties(50, 50, 400, 400) # position and size

fig_map.draw_vehicle(x0,size=1)

# Creating random map of landmarks
nb_landmarks = 10
map_area = IntervalVector(2,[-10,10])
fig_map.axis_limits(map_area.inflate(10))
v_map = DataLoader.generate_landmarks_boxes(map_area, nb_landmarks)
ctc_constell = myCtc(v_map) # constellation constraint
cn0={}
for t in np.arange(dt,tmax,dt):
    tdomain_online,tdomain_online_sub=Interval(0,t),Interval(t-dt,t)
    xt_ref=ref_evolution(x_ref.last_value(),u_ref.last_value(),dt)
    x_ref.set(xt_ref,t)
    u_ref.set(control(a,b,xt_ref),t)

    # Sets of trajectories
    x = TubeVector(tdomain_online, dt, 3)                    # 4d tube for state vectors
    v = TubeVector(tdomain_online, dt, 3)                    # 4d tube for derivatives of the states
    u = Tube(tdomain_online, dt)                    # tube for input of the system
    x[2] = Tube(x_ref[2], dt).inflate(5*math.pi/180)       # estimated_heading by control & motion evolution
  
    # Sets of observations

    # Generating observations obs=(t,range,bearing) of these landmarks from current center point 'x_ref' AND current borders 
    max_nb_obs = nb_landmarks // 2 # arbitrary number of detected landmarks, to be tuned
    visi_range = Interval() 
    visi_angle = Interval() # NON frontal sonar
    v_obs = DataLoader.generate_observations(x_ref, v_map, max_nb_obs, True, visi_range, visi_angle)

    # Adding uncertainties on the measurements
    for obs in v_obs:
        
        obs[1].inflate(1.5) # range inflated by 1.5m
        obs[2].inflate(0.1*math.pi/180) # bearing initially 0.1 deg
    

    # Association set
    m = [IntervalVector(2) for _ in v_obs] # unknown association

    # =========== CONTRACTOR NETWORK ===========

    cn = ContractorNetwork()
    
    cn.add(ctc_f, [v, x, u])   # adding the f constraint
    cn.add(ctc.deriv, [x, v])

    for i in range(0,len(v_obs)):
    
        # Measurement i
        ti  = Interval(v_obs[i][0]) # time
        y1 = Interval(v_obs[i][1]) # range
        y2 = Interval(v_obs[i][2]) # bearing

        # Intermediate variables:
        z = cn.create_interm_var(Interval())
        d = cn.create_interm_var(IntervalVector(2))
        p = cn.create_interm_var(IntervalVector(3))
        
        cn.add(ctc_constell, [m[i]])
        cn.add(ctc_minus, [d, m[i], cn.subvector(p,0,1)])
        cn.add(ctc_plus, [z, p[2], y2])
        cn.add(ctc.polar, [d, y1, z])
        cn.add(ctc.eval, [ti, p, x, v])

    cn.contract()

    # Update global x,u with specific interval (not feasible to modify previous estimations a posteriori, but still using past information)    
#    x_all.set(x(tdomain_online_sub),tdomain_online_sub)
#    u_all.set(u(tdomain_online_sub),tdomain_online_sub)#    if (x(t).subvector(0,1)).volume() < 20:

    if not (x(t).is_empty() and x(t).volume() < 20) : # not drawing outliers for the moment for readability
      fig_map.draw_box(x(t).subvector(0,1),'black[]')
    for i in range(0,len(v_obs)):
      if (not m[i].is_empty() and m[i].volume() < 20):
        fig_map.draw_box(m[i],'red[orange]')

fig_map.show()

beginDrawing()
fig_control=VIBesFigTube('Associated Control over time')
fig_control.set_properties(50, 50, 400, 400) # position and size
fig_control.add_trajectory(u_ref,"u*", "red")
fig_control.show()
endDrawing()
