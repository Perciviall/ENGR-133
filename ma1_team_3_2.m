%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course Number: ENGR 13300
% Semester: e.g. Fall 2024
%
% Problem Description: Add the problem description here and delete this
%                      line.
%
% Assignment Information
%   Assignment:     Team 3 - MA1
%   Author:         Milo Lamas, lamasm@purdue.edu
%                   Kevin Dong –  dong417@purdue.edu 
%                   Ravi Atheyra - athreyr@purdue.edu
%                   Abdullah Mohammed – mohame49@purdue.edu
%                   Ryan Faraji - rfaraji@purdue.edu 
%   Team ID:        021-02 (e.g. LC1 - 01; for section LC1, team 01)
%   Date:           10/22/2024
%
%   Contributor:    Name, login@purdue [repeat for each]
%   My contributor(s) helped me:
%     [ ] understand the assignment expectations without
%         telling me how they will approach it.
%     [ ] understand different ways to think about a solution
%         without helping me plan my solution.
%     [ ] think through the meaning of a specific error or
%         bug present in my code without looking at my code.
%   Note that if you helped somebody else with their code, you
%   have to list that person as a contributor here as well.
%
% Academic Integrity Statement:
%     I have not used source code obtained from any unauthorized
%     source, either modified or unmodified; nor have I provided
%     another student access to my code.  The project I am
%     submitting is my own original work.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ____________________
%% INITIALIZATION

%tank 1 diameter in m
%tank1_diameter = input('Enter diameter of Tank 1: ');
tank1_diameter = 12; 

%tank 1 height in m
%tank1_height = input('Enter height of Tank 1: ');
tank1_height = 5;

%tank 2 diameter in m
%tank2_diameter = input('Enter diameter of Tank 2: ');
tank2_diameter = 4; 

%tank 2 height in m
%tank2_height = input('Enter height of Tank 2: ');
tank2_height = 9; 


%% ____________________
%% CALCULATIONS

% tank 1 volume in m^3
tank1_volume_meterscubed = (pi)*((tank1_diameter/2)^2)*(tank1_height);
% tank 1 volume in gallons
tank1_capacity = (tank1_volume_meterscubed)*(264.172);

% tank 2 volume in m^3
tank2_volume_meterscubed = (pi)*((tank2_diameter/2)^2)*(tank2_height);
% tank 2 volume in gallons
tank2_capacity = (tank2_volume_meterscubed)*(264.172); 

%% ____________________
%% OUTPUTS

% Display tank 1 capacity using disp and variable call
disp('The capacity of Tank 1 in U.S. gallons is:')
tank1_capacity

%Display tank 2 capacity, diameter, and height using two fprintf statements
fprintf('The capacity of Tank 2 is %.0f U.S. gallons.\n', tank2_capacity)
fprintf('Tank 2 has a diameter of %.1f ft and is %.1f. ft tall.\n', (tank2_diameter* 3.28084), (tank2_height* 3.28084))

%% ____________________

% Thanks! Have a great day! :)