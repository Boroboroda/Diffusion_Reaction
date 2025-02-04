% Definition of Diffusion Reaction function
function [c, f, s] = pdefun(x, t, u, dudx)
    N_sa = 90;
    N_sb = 48;
    Annealing_time =60; %Options: 0.1, 10, 60
    CA = u(1);
    CBC = u(2);
    CC = u(3);
    CAB = u(4);
    CABB = u(5);
    CAAB = u(6);

    %Apply Nondiensionalizaion on FDM
    Csum_1 = CA + CBC + CC + CAB + CABB + CAAB;
    Csum_2 = CBC + CC + CAB + CABB + CAAB;  
    D_A = 360*(Annealing_time/(600^2));  %360
    D_star = D_A *(Csum_2 / Csum_1);
    
    c = ones(6, 1); 

    % Effective heterogeneous diffusion rate
    f = [D_star * dudx(1); 
         D_star * dudx(2);
         D_star * dudx(3);
         0;
         0;
         0];

    % Reaction parameter
    k11=0; %Option: 3.6e-3, 0
    k12=0; %Option: 3.6e-3, 0
    k21=0;

    s = [-k11*(N_sa * N_sb * Annealing_time )/N_sa *u(1)*u(2) - k21*(N_sa * N_sa * Annealing_time )/N_sa*u(1)*u(4);
        -k11*(N_sa * N_sb * Annealing_time )/N_sb*u(1)*u(2) - k12*(N_sa * N_sb * Annealing_time )/N_sb*u(2)*u(4);
        k11*(N_sa * N_sb * Annealing_time )/N_sa*u(1)*u(2) + k12*(N_sa * N_sb * Annealing_time )/N_sa*u(2)*u(4);
        k11*(N_sa * N_sb * Annealing_time )/N_sa*u(1)*u(2) - k21*(N_sa * N_sb * Annealing_time )/N_sa*u(1)*u(4) - k12*(N_sa * N_sb * Annealing_time )/N_sa*u(2)*u(4);
        k12*(N_sa * N_sb * Annealing_time )/N_sa*u(2)*u(4);
        k21*(N_sa * N_sb * Annealing_time )/N_sa*u(1)*u(4)];

    % s = [-k11*u(1)*u(2) - k21*u(1)*u(4);b
    %     -k11*u(1)*u(2) - k12*u(2)*u(4);
    %     k11*u(1)*u(2) + k12*u(2)*u(4);
    %     k11*u(1)*u(2) - k21*u(1)*u(4) - k12*u(2)*u(4);
    %     k12*u(2)*u(4);
    %     k21*u(1)*u(4)];
end

% Initial Condition
function u0 = icfun(x)
    N_SA = 90; 
    N_SB = 48;
    h= 100; 
    L=600;
    if x <= h/L
        u0 = [1; 0; 0; 0; 0; 0];
    else
        u0 = [0; 1; 0; 0; 0; 0];
    end
end

% Boundary Condition
function [pl, ql, pr, qr] = bcfun(xl, ul, xr, ur, t)
    % Left x = 0
    pl = zeros(6,1);  % 左边界条件
    ql = ones(6,1);
    
    % Right x = L
    pr = zeros(6,1);  % 右边界条件
    qr = ones(6, 1);
end

% Main function, Use pdepe to get result
function main
    m = 0; 
    mesh_gap = 1001;
    x = linspace(0, 1, mesh_gap);  % spatial
    t = linspace(0, 1, mesh_gap);  % temporal

    sol = pdepe(m, @pdefun, @icfun, @bcfun, x, t);

    % u1 - u6
    u1 = sol(:,:,1)*90;
    u2 = sol(:,:,2)*48;
    u3 = sol(:,:,3)*90;
    u4 = sol(:,:,4)*90;
    u5 = sol(:,:,5)*90;
    u6 = sol(:,:,6)*90;
    u_sum = u1+u2+u3+u4+u5+u6;
    % disp(u1(100,:))
    charicteristic_concentration = {90, 48, 90, 90, 90, 90};
    % save
    filenames = {'u1_output.csv', 'u2_output.csv', 'u3_output.csv', 'u4_output.csv', 'u5_output.csv', 'u6_output.csv'};
    for i = 1:6
        u = (sol(:,:,i) * charicteristic_concentration{i})'; 
        data = [NaN, t; x(:), u];  
        filename = filenames{i}; 
        writematrix(data, filename);  
    end

    re_u1 = u1(mesh_gap,:) ./ u_sum(mesh_gap,:) *100;
    re_u2 = u2(mesh_gap,:) ./ u_sum(mesh_gap,:) * 1/2 *100;
    re_u3 = u3(mesh_gap,:) ./ u_sum(mesh_gap,:) *100;
    re_u4 = u4(mesh_gap,:) ./ u_sum(mesh_gap,:) *100;
    re_u5 = u5(mesh_gap,:) ./ u_sum(mesh_gap,:) *100;
    maxvalue = [max(u1(:)),max(u2(:)),max(u3(:)),max(u4(:)),max(u5(:)),max(u6(:))];
    disp(maxvalue)
    % disp(re_u2);
    % 

    % Plot figures
    figure;
    subplot(3, 2, 1);
    surf(x, t, u1);
    zlim([0,100])
    title('Numerical solution for Ni');
    xlabel('Distance x');
    ylabel('Time t');
    zlabel('Concentration Ni');

    subplot(3, 2, 2);
    surf(x, t, u2);
    zlim([0,100])
    title('Numerical solution for SiC');
    xlabel('Distance x');
    ylabel('Time t');
    zlabel('Concentration SiC');

    subplot(3, 2, 3);
    surf(x, t, u3);
    zlim([0,100])
    title('Numerical solution for C');
    xlabel('Distance x');
    ylabel('Time t');
    zlabel('Concentration C');

    subplot(3, 2, 4);
    surf(x, t, u4);
    zlim([0,100])
    title('Numerical solution for NiSi');
    xlabel('Distance x');
    ylabel('Time t');
    zlabel('Concentration NiSi');

    subplot(3, 2, 5);
    surf(x, t, u5);
    zlim([0,100])
    title('Numerical solution for NiSi2');
    xlabel('Distance x');
    ylabel('Time t');
    zlabel('Concentration NiSi2');

    subplot(3, 2, 6);
    plot(x,re_u1);
    hold on;
    plot(x,re_u2)
    hold on;
    plot(x,re_u3)
    hold on;
    plot(x,re_u4)
    hold on;
    plot(x,re_u5)
    title('Numerical solution of Relative Concentration');
    legend("Ni","SiC","C","NiSi","NiSi2")
    ylim([0,100])
    xlabel('Distance x');
    ylabel('Relative Concentration');
end

% Run main
main
