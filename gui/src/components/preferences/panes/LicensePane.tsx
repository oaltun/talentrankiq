import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Key, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { invoke } from '@tauri-apps/api/core'

type LicenseStatus =
  | {
      kind: 'licensed'
      customer: string
      expires_at: string
      artifacts: string[]
    }
  | { kind: 'invalid'; reason: string }
  | { kind: 'not_found' }
  | { kind: 'revoked'; reason: string }

export function LicensePane() {
  const { t } = useTranslation()
  const [status, setStatus] = useState<LicenseStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [licenseKey, setLicenseKey] = useState('')
  const [activating, setActivating] = useState(false)
  const [message, setMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  const refreshStatus = async () => {
    setLoading(true)
    try {
      const result = await invoke<LicenseStatus>('license_status_full')
      setStatus(result)
    } catch (e) {
      setStatus({ kind: 'invalid', reason: String(e) })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!loading && status === null) {
      refreshStatus()
    }
  }, [loading, status])

  const handleActivate = async () => {
    if (!licenseKey.trim()) return
    setActivating(true)
    setMessage(null)
    try {
      const result = await invoke<LicenseStatus>('license_activate', {
        licenseKey: licenseKey.trim(),
      })
      if (result.kind === 'licensed') {
        setMessage({
          type: 'success',
          text: t('preferences.license.activated'),
        })
        setLicenseKey('')
      } else if (result.kind === 'invalid') {
        setMessage({
          type: 'error',
          text: t('preferences.license.invalid', { reason: result.reason }),
        })
      } else if (result.kind === 'revoked') {
        setMessage({
          type: 'error',
          text: t('preferences.license.revoked', { reason: result.reason }),
        })
      }
      setStatus(result)
    } catch (e) {
      setMessage({ type: 'error', text: String(e) })
    } finally {
      setActivating(false)
    }
  }

  const handleDeactivate = async () => {
    setActivating(true)
    setMessage(null)
    try {
      await invoke('license_deactivate')
      setMessage({
        type: 'success',
        text: t('preferences.license.deactivated'),
      })
      setStatus({ kind: 'not_found' })
    } catch (e) {
      setMessage({ type: 'error', text: String(e) })
    } finally {
      setActivating(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">
          {t('preferences.license.title')}
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          {t('preferences.license.description')}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            {t('preferences.license.status')}
          </CardTitle>
          <CardDescription>
            {t('preferences.license.statusDescription')}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {loading ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{t('preferences.license.checking')}</span>
            </div>
          ) : status ? (
            <>
              {status.kind === 'licensed' && (
                <div className="flex items-center gap-3 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-green-700 dark:text-green-400">
                      {t('preferences.license.licensedTo', {
                        customer: status.customer,
                      })}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {t('preferences.license.expires', {
                        date: new Date(status.expires_at).toLocaleDateString(),
                      })}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {t('preferences.license.artifacts', {
                        artifacts: status.artifacts.join(', '),
                      })}
                    </p>
                  </div>
                </div>
              )}
              {status.kind === 'not_found' && (
                <div className="flex items-center gap-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <AlertCircle className="h-5 w-5 text-yellow-500 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-yellow-700 dark:text-yellow-400">
                      {t('preferences.license.notFound')}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {t('preferences.license.notFoundDescription')}
                    </p>
                  </div>
                </div>
              )}
              {status.kind === 'invalid' && (
                <div className="flex items-center gap-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <XCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-red-700 dark:text-red-400">
                      {t('preferences.license.invalid')}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {status.reason}
                    </p>
                  </div>
                </div>
              )}
              {status.kind === 'revoked' && (
                <div className="flex items-center gap-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <XCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-red-700 dark:text-red-400">
                      {t('preferences.license.revoked')}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {status.reason}
                    </p>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-muted-foreground">
              {t('preferences.license.unknown')}
            </div>
          )}

          <Separator />

          <div className="space-y-3">
            <h4 className="font-medium">
              {t('preferences.license.activateLicense')}
            </h4>
            <p className="text-sm text-muted-foreground">
              {t('preferences.license.enterKey')}
            </p>
            <div className="flex gap-2">
              <Input
                placeholder="TRIQ-XXXX-XXXX-XXXX"
                value={licenseKey}
                onChange={e => setLicenseKey(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleActivate()}
                disabled={activating}
                className="flex-1 font-mono"
              />
              <Button
                onClick={handleActivate}
                disabled={activating || !licenseKey.trim()}
              >
                {activating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    {t('preferences.license.activating')}
                  </>
                ) : (
                  t('preferences.license.activate')
                )}
              </Button>
            </div>

            {status && status.kind === 'licensed' && (
              <Button
                variant="destructive"
                onClick={handleDeactivate}
                disabled={activating}
              >
                {activating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    {t('preferences.license.deactivating')}
                  </>
                ) : (
                  t('preferences.license.deactivate')
                )}
              </Button>
            )}

            {message && (
              <div
                className={`text-sm p-2 rounded ${
                  message.type === 'success'
                    ? 'bg-green-500/10 text-green-700 dark:text-green-400'
                    : 'bg-red-500/10 text-red-700 dark:text-red-400'
                }`}
              >
                {message.text}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t('preferences.license.howItWorks')}</CardTitle>
          <CardDescription>
            {t('preferences.license.howItWorksDescription')}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <ul className="list-disc list-inside space-y-1">
            <li>{t('preferences.license.feature1')}</li>
            <li>{t('preferences.license.feature2')}</li>
            <li>{t('preferences.license.feature3')}</li>
            <li>{t('preferences.license.feature4')}</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
