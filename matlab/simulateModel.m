clear all
close all
clc


dt = 0.01; % Computation time step
h_b_desired = 1.2; % Desired apex height of the bounce
initialBounce = 1; % Set to 1 for initial bounce, later changed to 0

x0_ee = -0.3; y0_ee = 0.4; % Coordinates of initial position of ee
[q1_temp, q2_temp, q3_temp] = inv_kin(x0_ee, y0_ee, pi); % Finding initial joint angles
q_initial(1,1) = q1_temp(1,1); q_initial(1,2) = q2_temp(1,1); q_initial(1,3) = q3_temp(1,1);
q_initial(1,:) = wrapToPi( [(q_initial(1,1)+3*pi/2) q_initial(1,2) q_initial(1,3)] );

b_initial(1,1) = 0; b_initial(1,2) = 1.2; % Initial position of the ball
dq_initial(1,1) = 0; dq_initial(1,2) = 0; dq_initial(1,3) = 0; % Initial joint velocities
b_dot_initial(1,1) = 0; b_dot_initial(1,2) = 0; % x and y coordinates of initial ball velocity

X0 = [q_initial b_initial dq_initial b_dot_initial]'; % Initial state vector
curX = X0;


n = 4; % Number of bounces
for i = 1:n
    replayX = [];
    [t, X_desired, ee_desired, ee_dot_desired, error_in_final_desired] = planning(X0, h_b_desired, dt, initialBounce);
    curX  = X_desired(1,:)'; 
    for j = 2:size(X_desired,1)

        replayX = [replayX curX];

        X_des = X_desired(j, :)';
        U = controller(X_des,curX);
        [tout, xout] = integrateODE([0 dt], curX, dt, U);
        [preImpactState, impactTime] = detectImpact(tout,xout);
        if impactTime ~= -1
            curX = impact(preImpactState');
            initialBounce = 0;
            X0 = curX;
            %break
        else
            curX = xout(2,:)';
        end

     
    end
    animate(1:1:size(replayX,2), replayX');
    clf
   

end