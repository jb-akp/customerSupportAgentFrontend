import type { AppConfig } from './lib/types';

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'iPear',
  pageTitle: 'iPear Customer Support',
  pageDescription: 'Get help from our AI assistant',

  supportsChatInput: true,
  supportsVideoInput: true,
  supportsScreenShare: false,
  isPreConnectBufferEnabled: true,

  logo: '/ipear-logo.svg',
  accent: '#007AFF',
  logoDark: '/ipear-logo-dark.svg',
  accentDark: '#0A84FF',
  startButtonText: 'Start Support Session',
};
