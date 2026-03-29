import { z } from 'zod';

export const SuggestionSchema = z.object({
  SN: z.number().optional(),
  Gid: z.string(),
  Name: z.string(),
  Title: z.string(),
  Department: z.string(),
  Tel: z.string(),
  LaborType: z.string(),
  Shift: z.string(),
  UserAreaName: z.string(),
  ProblemAreaName: z.string(),
  LocationName: z.string(),
  ManagerGid: z.string(),
  ManagerName: z.string(),
  SubmissionDate: z.string().optional(),
  LeanFlowName: z.string().optional(),
  Description: z.string(),
  Suggestion: z.string(),
  ReplyOpinion: z.string().optional(),
  RejectJustification: z.string().optional(),
  Status: z.string().optional(),
  ItemType: z.string(),
  GoodRequest: z.string().optional(),
  OwnerGid: z.string(),
  OwnerName: z.string(),
  OwnerTel: z.string(),
  OwnerManagerGid: z.string(),
  OwnerManagerName: z.string(),
  Score: z.number().optional(),
  IsRepeat: z.string(),
  AreaType: z.string(),
});

export type Suggestion = z.infer<typeof SuggestionSchema>;

export const InputFields = ['Description', 'Suggestion', 'Gid'] as const;
export type InputField = (typeof InputFields)[number];

export const OutputFields = [
  'Title',
  'Department',
  'LaborType',
  'Shift',
  'UserAreaName',
  'ProblemAreaName',
  'LocationName',
  'ManagerGid',
  'ManagerName',
  'ItemType',
  'OwnerName',
  'OwnerGid',
  'OwnerTel',
  'OwnerManagerGid',
  'OwnerManagerName',
  'IsRepeat',
  'AreaType',
] as const;

export type OutputField = (typeof OutputFields)[number];

export const PredictionResult = SuggestionSchema.pick({
  Title: true,
  Department: true,
  LaborType: true,
  Shift: true,
  UserAreaName: true,
  ProblemAreaName: true,
  LocationName: true,
  ManagerGid: true,
  ManagerName: true,
  ItemType: true,
  OwnerName: true,
  OwnerGid: true,
  OwnerTel: true,
  OwnerManagerGid: true,
  OwnerManagerName: true,
  IsRepeat: true,
  AreaType: true,
});

export type PredictionResult = z.infer<typeof PredictionResult>;
