function [preImpactState, impactTime] = detectImpact(tout,xout)

stateDimension = length(xout(1,:));
preImpactState = zeros(stateDimension,1,'single');
impactTime = -1; % if this is unchanged implies no impact

params;

q1 = xout(:,1); q2 = xout(:,2); q3 = xout(:,3);
xb = xout(:,4); yb = xout(:,5);

th1 = q1;
th2 = q2 + q1;
th3 = q3 + q2 + q1;

x1 =  -l1*sin(th1);
x2 = x1 - l2*sin(th2);
x3 = x2 - l3*sin(th3);
y1 = l1*cos(th1);
y2 = y1 + l2*cos(th2);
y3 = y2 + l3*cos(th3);

impLinkSlope = (y3 - y2)./(x3 - x2);
impLinkIntercept = y3 - impLinkSlope.*x3;

%% Detecting impacts
distanceToImpLink = abs(yb - xb.*impLinkSlope - impLinkIntercept)./sqrt(1 + impLinkSlope.^2);
 if (distanceToImpLink(2) < rb)
     if ((x2(2) < xb(2) && xb(2) < x3(2)) || (x3(2) < xb(2) && xb(2) < x2(2)))
         impactTime = interp1([distanceToImpLink(1) distanceToImpLink(2)], tout, rb,'PCHIP');
         preImpactState = interp1(tout,xout,impactTime,'PCHIP');
    end
end


end