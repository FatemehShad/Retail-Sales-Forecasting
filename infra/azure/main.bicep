targetScope = 'resourceGroup'

@description('Globally unique, lowercase ADLS Gen2 account name.')
@minLength(3)
@maxLength(24)
param storageAccountName string
param location string = resourceGroup().location
param filesystemName string = 'retail'
param accessConnectorName string = 'retail-databricks-connector'
param dataFactoryName string

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    isHnsEnabled: true
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    accessTier: 'Hot'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: filesystemName
  properties: { publicAccess: 'None' }
}

resource connector 'Microsoft.Databricks/accessConnectors@2023-05-01' = {
  name: accessConnectorName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {}
}

resource blobContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, connector.id, 'blob-contributor')
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalId: connector.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource factory 'Microsoft.DataFactory/factories@2018-06-01' = {
  name: dataFactoryName
  location: location
  identity: { type: 'SystemAssigned' }
}

output rawDataUri string = 'abfss://${filesystemName}@${storage.name}.dfs.${environment().suffixes.storage}/kaggle/store-sales'
output accessConnectorId string = connector.id
output dataFactoryPrincipalId string = factory.identity.principalId
