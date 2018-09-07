MyL1Criterion, parent = torch.class('nn.MyL1Criterion', 'nn.Criterion')

function MyL1Criterion:__init()
   parent.__init(self)
end

function MyL1Criterion:updateOutput(input, target)

   -- loss is the Euclidean distance between predicted and ground truth coordinate, mean calculated over batch
   local dists = torch.norm(input - target, 2, 2)
   -- set loss to zero if target element is invalid (all 0s)
   for i=1, target:size(1) do
    if target[i][1]<1e-8 and target[i][2]<1e-8 and target[i][3]<2e-8 then
     dists[i]:zero()
    end
   end
   self.output = torch.mean(dists)

   return self.output
end

function MyL1Criterion:updateGradInput(input, target)
   -- gradients are the difference of predicted and ground truth coordinate divided (scaled) by the Euclidean distance
   local dists = torch.norm(input - target, 2, 2)
   dists = torch.expand(dists, dists:size(1), 3)
   self.gradInput = torch.cdiv(input-target,dists)

   -- set gradient to zero if target element is invalid (all 0s)
   for i=1, target:size(1) do
    if target[i][1]<1e-8 and target[i][2]<1e-8 and target[i][3]<2e-8 then
     self.gradInput[i]:zero()
    end
   end

   self.gradInput = torch.div(self.gradInput, dists:size(1))
   return self.gradInput
end
