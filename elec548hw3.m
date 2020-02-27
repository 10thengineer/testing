load ('ReachData.mat');
%%
%set up basic parameters
num_trials= 1127;
num_neurons=190;
num_directions = 8;
num_training_trials=50;
num_test_trials=num_trials-(num_directions*num_training_trials);

%Set up times in arrays
for x = 1:length(r)
    timeTouchHeld(x,1)=r(x).timeTouchHeld;
    timeGoCue(x,1)=r(x).timeGoCue;
    timeTargetAcquire(x,1)=r(x).timeTargetAcquire;

end
%set up start and stop points
plan_window_time=[timeTouchHeld'; timeGoCue'];
mvm_window_time=[timeGoCue'; timeTargetAcquire'];
combined_window_time=[timeTouchHeld'; timeTargetAcquire'];

%set up firing window length
plan_window_length=(diff(plan_window_time)/1000);
mvm_window_length=(diff(mvm_window_time)/1000);
combined_window_length=(diff(combined_window_time)/1000);

%initialize matrices
p_count=zeros(num_trials,num_neurons);
m_count=zeros(num_trials,num_neurons);
c_count=zeros(num_trials,num_neurons);

%count spikes in windows and normalize to count per second
for trial= 1:num_trials
    for neuron = 1:num_neurons
        spikes= r(trial).unit(neuron).spikeTimes;
        p_count(trial,neuron)= sum((plan_window_time(1,trial)<=spikes)&(spikes<=plan_window_time(2,trial)))/plan_window_length(trial);
        m_count(trial,neuron)= sum((mvm_window_time(1,trial)<=spikes)&(spikes<=mvm_window_time(2,trial)))/mvm_window_length(trial);
        c_count(trial,neuron)= sum((combined_window_time(1,trial)<=spikes)&(spikes<=combined_window_time(2,trial)))/combined_window_length(trial);
    end
end
%create structure for normalized counts per second
training_trials=zeros(num_directions,num_training_trials);
%create structure for mean 
p_ave=zeros(num_neurons,num_directions);
m_ave=zeros(num_neurons,num_directions);
c_ave=zeros(num_neurons,num_directions);

%% iterate through directions and get estimates for mean
for directions = 1:num_directions
    same_direction_trials=find(cfr==directions);
    training_trials(directions,:)=datasample(same_direction_trials,num_training_trials,'Replace',false);%random unique training trial indices 
    %^array of indeces of 50 trials for each direction used for training
    %estimate parameters
    %get mean for each trial: mean firing rate for each neuron for each
    %direction according to training trial data
    p_ave(:,directions)=mean(p_count(training_trials(directions,:),:),1); 
    m_ave(:,directions)=mean(m_count(training_trials(directions,:),:),1);
    c_ave(:,directions)=mean(c_count(training_trials(directions,:),:),1);

end

%replace 0's with small numbers
p_ave(p_ave==0)=eps;
m_ave(m_ave==0)=eps;
c_ave(c_ave==0)=eps;

%array initialization
pois_plan=zeros(num_test_trials,num_directions);
pois_move=zeros(num_test_trials,num_directions);
pois_comb=zeros(num_test_trials,num_directions);
test_directions=zeros(num_test_trials,1);
training_trials_vec= reshape(training_trials, [1,num_directions*num_training_trials]);%formatting for ismember function
%%

%test trials
test_trial_idx=1;
for trial=1:num_trials 
    if ~ismember(trial,training_trials_vec)%is not member function
        %retrieve test directions
        test_directions(test_trial_idx)=cfr(trial);
         %firing rate counts for each neuron
      
         p_count_test=p_count(trial,:)';
         m_count_test=m_count(trial,:)';
         c_count_test=c_count(trial,:)';
    %%

    %compute log likelihood for each direction
         for directions=1:num_directions
             pois_plan(test_trial_idx,directions)=sum((p_count_test.*log(p_ave(:,directions)*plan_window_length(1,trial)))-(p_ave(:,directions)*plan_window_length(1,trial)));
             pois_move(test_trial_idx,directions)=sum((m_count_test.*log(m_ave(:,directions)*mvm_window_length(1,trial)))-(m_ave(:,directions)*mvm_window_length(1,trial)));
             pois_comb(test_trial_idx,directions)=sum((c_count_test.*log(c_ave(:,directions)*combined_window_length(1,trial)))-(c_ave(:,directions)*combined_window_length(1,trial)));

         end
    test_trial_idx=test_trial_idx+1;
    %increment index
    end
    

end
[maxp_val,maxp_ind]=max(pois_plan,[],2);
[maxm_val,maxm_ind]=max(pois_move,[],2);
[maxc_val,maxc_ind]=max(pois_comb,[],2);
%calculate accuracy
pp_acc= 100*sum(maxp_ind==test_directions)/num_test_trials
pm_acc= 100*sum(maxm_ind==test_directions)/num_test_trials
pc_acc= 100*sum(maxc_ind==test_directions)/num_test_trials
