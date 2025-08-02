import { Button } from '@/components/ui/button';

interface WelcomeProps {
  disabled: boolean;
  startButtonText: string;
  onStartCall: () => void;
}

export const Welcome = ({
  disabled,
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeProps) => {
  return (
    <div
      ref={ref}
      inert={disabled}
      className="fixed inset-0 z-10 mx-auto flex h-svh flex-col items-center justify-center text-center bg-slate-900"
    >
      {/* iPear Logo */}
      <div className="mb-8">
        <div className="w-16 h-16 mx-auto mb-6 bg-blue-500 rounded-2xl flex items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M24 4C26.2091 4 28 5.79086 28 8C28 10.2091 26.2091 12 24 12C21.7909 12 20 10.2091 20 8C20 5.79086 21.7909 4 24 4Z" fill="white"/>
            <path d="M32 16C32 13.7909 30.2091 12 28 12H20C17.7909 12 16 13.7909 16 16V36C16 38.2091 17.7909 40 20 40H28C30.2091 40 32 38.2091 32 36V16Z" fill="white"/>
            <path d="M22 20C22 18.8954 22.8954 18 24 18C25.1046 18 26 18.8954 26 20V32C26 33.1046 25.1046 34 24 34C22.8954 34 22 33.1046 22 32V20Z" fill="#007AFF"/>
          </svg>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-sm mx-auto px-6">
        <h1 className="text-2xl font-semibold text-white mb-4">
          iPear Support
        </h1>
        
        <p className="text-slate-300 mb-8">
          Get help from our AI assistant
        </p>

        <Button 
          variant="primary" 
          size="lg" 
          onClick={onStartCall} 
          className="w-full h-12 text-lg font-medium bg-blue-500 hover:bg-blue-600 text-white rounded-xl"
        >
          {startButtonText}
        </Button>
      </div>
    </div>
  );
};
